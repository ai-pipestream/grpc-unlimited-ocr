// grpc-unlimited-ocr: the gRPC collector. Loads the model once, then
// serves ParseDocument and GetServiceInfo on GRPC_UOCR_LISTEN_ADDRESS
// (default 0.0.0.0:50073). Diskless while parsing: pages, tensors and
// output stay in memory; only the read-only models directory is read.

#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <print>
#include <stdexcept>
#include <thread>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "service/config.h"
#include "service/model_engine.h"
#include "service/unlimited_ocr_service.h"

namespace {

// The handler only writes a byte to a pipe: grpc::Server::Shutdown is not
// async-signal-safe, so a thread does the real shutdown.
int g_shutdown_pipe[2] = {-1, -1};

void handle_shutdown(int /*signum*/) {
  const char one = 1;
  [[maybe_unused]] const ssize_t n = ::write(g_shutdown_pipe[1], &one, 1);
}

void install_shutdown_pipe() {
  if (::pipe2(g_shutdown_pipe, O_CLOEXEC) != 0) throw std::runtime_error("cannot create the shutdown pipe");
  struct sigaction action = {};
  action.sa_handler = handle_shutdown;
  ::sigaction(SIGINT, &action, nullptr);
  ::sigaction(SIGTERM, &action, nullptr);
  ::signal(SIGPIPE, SIG_IGN);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  try {
    install_shutdown_pipe();
    const uocr::service::Config config = uocr::service::load_config_from_env();
    const auto t0 = std::chrono::steady_clock::now();
    const std::unique_ptr<uocr::service::Engine> engine = uocr::service::load_model_engine(config);
    std::println("model loaded from {} on {} ({}) in {:.1f} s", config.models_dir, engine->info().device,
                 config.weight_format, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    uocr::service::UnlimitedOcrServiceImpl service(config, *engine);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(config.listen_address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(static_cast<int>(config.max_page_bytes + 1024 * 1024));
    builder.RegisterService(&service);
    const std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (server == nullptr) {
      std::println(stderr, "cannot listen on {}", config.listen_address);
      return 1;
    }
    std::thread shutdown_thread([&server] {
      char byte = 0;
      [[maybe_unused]] const ssize_t n = ::read(g_shutdown_pipe[0], &byte, 1);
      server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(10));
    });
    std::println("grpc-unlimited-ocr listening on {} (backend {}, window {} pages, {} document(s) in flight)",
                 config.listen_address, config.backend, config.page_window, config.documents_in_flight);
    server->Wait();
    handle_shutdown(0);
    shutdown_thread.join();
    std::println("documents ok {}, failed {}", service.documents_ok.load(), service.documents_failed.load());
    return 0;
  } catch (const std::exception& error) {
    std::println(stderr, "startup failed: {}", error.what());
    return 1;
  }
}
