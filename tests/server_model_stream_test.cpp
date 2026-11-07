#include <grpcpp/grpcpp.h>
// Real weights through the gRPC transport, with one and two pages, a
// slow reader and cancellation. Never silently substitutes a fake engine.
#include <chrono>
#include <thread>
#include "check.h"
#include "backend/reference/parallel.h"
#include "output_helpers.h"
#include "service/model_engine.h"
#include "service_helpers.h"

using namespace uocr;
using namespace uocr::testing;
using namespace std::chrono_literals;

int main(int argc, char** argv) {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) return skip("UOCR_MODELS_DIR is not set");
  return run_tests([&] {
    // Keep this optional real-model test from consuming every host CPU.
    reference::set_thread_count(2);
    ::setenv("GRPC_UOCR_MODELS_DIR", models, 1);
#if !UOCR_WITH_CUDA
    ::setenv("GRPC_UOCR_BACKEND", "reference", 0);
#endif
    auto config = service::load_config_from_env();
    config.page_window = 2;
    config.max_pages = 2;
    config.max_tokens_per_page = 600;
    const auto engine = service::load_model_engine(config);
    service::UnlimitedOcrServiceImpl service(config, *engine);
    grpc::ServerBuilder builder;
    builder.RegisterService(&service);
    const auto server = builder.BuildAndStart();
    const auto stub = v1::UnlimitedOcrService::NewStub(server->InProcessChannel({}));
    grpc::ClientContext info_context;
    v1::GetServiceInfoResponse info;
    require(stub->GetServiceInfo(&info_context, {}, &info).ok(), "real model service info");
    require(info.backend() == config.backend, "reported backend matches loaded backend");
    require(info.weight_format() == (config.backend == "reference" ? "fp32" : config.weight_format),
            "reported precision matches loaded backend");
    if (argc == 2 && std::string(argv[1]) == "--info-only") {
      std::println("real model: backend={}, precision={}, device={}", info.backend(), info.weight_format(), info.device());
      server->Shutdown();
      return;
    }
    const auto png = read_text(std::string(UOCR_GOLDENS) + "/quickstart-p2/page.png");
    for (const int pages : {1, 2}) {
      std::println("starting {} page(s): backend={}, precision={}", pages, info.backend(), info.weight_format());
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + 20min);
      auto stream = stub->ParseDocument(&context);
      require(stream->Write(options_request(static_cast<std::uint32_t>(pages))), "options sent");
      for (int page = 1; page <= pages; ++page) {
        require(stream->Write(page_request(static_cast<std::uint32_t>(page), png)), "raster sent");
      }
      stream->WritesDone();
      v1::ParseDocumentResponse event;
      int token_pages = 0;
      int fragments = 0;
      bool complete = false;
      while (stream->Read(&event)) {
        if (event.has_page_failure()) throw std::runtime_error(event.page_failure().detail());
        if (event.has_tokens()) {
          ++token_pages;
          require(event.tokens().steps_size() > 0, "real decode produced tokens");
        }
        fragments += event.has_page_document() ? 1 : 0;
        if (event.has_complete()) {
          complete = true;
          require(event.complete().pages_ok() == static_cast<unsigned>(pages), "all pages successful");
        }
        // Consume deliberately slowly without withholding input needed for prefill.
        std::this_thread::sleep_for(10ms);
      }
      require(stream->Finish().ok() && complete && token_pages == pages && fragments == pages,
              "complete real-model page stream");
      std::println("{}: {} page(s), tokens and fragments complete", engine->info().backend, pages);
    }
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 20min);
    auto stream = stub->ParseDocument(&context);
    require(stream->Write(options_request(1)) && stream->Write(page_request(1, png)), "cancel request sent");
    stream->WritesDone();
    v1::ParseDocumentResponse event;
    require(stream->Read(&event) && event.has_window_started(), "real prefill completed");
    context.TryCancel();
    while (stream->Read(&event)) {}
    require(stream->Finish().error_code() == grpc::StatusCode::CANCELLED, "real-model stream cancelled");
    server->Shutdown(std::chrono::system_clock::now() + 10s);
    server->Wait();
    require(service.buffered_bytes().in_use() == 0 && service.document_slots().in_use() == 0,
            "real-model cancellation returned leases");
  });
}
