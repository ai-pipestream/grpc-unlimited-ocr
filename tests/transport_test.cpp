#include <grpcpp/grpcpp.h>
#include <thread>
#include "check.h"
#include "service_helpers.h"

using namespace uocr;
using namespace uocr::testing;
using namespace std::chrono_literals;

struct Harness {
  service::Config config;
  FakeEngine engine;
  std::unique_ptr<service::UnlimitedOcrServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<v1::UnlimitedOcrService::Stub> stub;
  Harness() { config.transport_idle_ms = 150; config.page_window = 1; }
  void start() {
    service = std::make_unique<service::UnlimitedOcrServiceImpl>(config, engine);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service.get());
    server = builder.BuildAndStart();
    require(server && port != 0, "TCP test server started");
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    stub = v1::UnlimitedOcrService::NewStub(grpc::CreateCustomChannel(
        "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials(), args));
  }
  bool released() {
    const auto until = std::chrono::steady_clock::now() + 2s;
    while (service->document_slots().in_use() && std::chrono::steady_clock::now() < until) {
      std::this_thread::sleep_for(5ms);
    }
    return service->document_slots().in_use() == 0 && service->buffered_bytes().in_use() == 0;
  }
  void recovery() {
    require(released(), "prior request released budgets");
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 3s);
    auto call = stub->ParseDocument(&context);
    require(call->Write(options_request(1)) && call->Write(page_request(1)), "recovery request sent");
    v1::ParseDocumentResponse event;
    bool complete = false;
    while (call->Read(&event)) complete |= event.has_complete();
    require(call->Finish().ok() && complete, "recovery succeeds without client half-close");
  }
  ~Harness() {
    if (server) { server->Shutdown(std::chrono::system_clock::now() + 1s); server->Wait(); }
  }
};

void idle(bool with_options) {
  Harness h;
  h.engine.windows = {{simple_page(1)}};
  h.start();
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 3s);
  auto call = h.stub->ParseDocument(&context);
  if (with_options) require(call->Write(options_request(1)), "options sent");
  const auto began = std::chrono::steady_clock::now();
  v1::ParseDocumentResponse event;
  while (call->Read(&event)) {}
  const auto status = call->Finish();
  require(status.error_code() == grpc::StatusCode::CANCELLED, "watchdog cancellation is the actual wire status");
  require(std::chrono::steady_clock::now() - began < 2s, "idle transport terminates before client deadline");
  h.recovery();
}

void admission() {
  Harness h;
  h.config.transport_idle_ms = 5000;
  h.engine.windows = {{simple_page(1)}};
  h.start();
  grpc::ClientContext first;
  first.set_deadline(std::chrono::system_clock::now() + 3s);
  auto a = h.stub->ParseDocument(&first);
  const auto until = std::chrono::steady_clock::now() + 500ms;
  while (!h.service->document_slots().in_use() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(5ms);
  const bool admitted_before_options = h.service->document_slots().in_use() == 1;
  grpc::ClientContext second;
  second.set_deadline(std::chrono::system_clock::now() + 2s);
  auto b = h.stub->ParseDocument(&second);
  v1::ParseDocumentResponse event;
  while (b->Read(&event)) {}
  const auto status = b->Finish();
  first.TryCancel();
  while (a->Read(&event)) {}
  a->Finish();
  require(admitted_before_options, "admission precedes initial Read");
  require(status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "excess RPC fails before reading or workers");
  h.recovery();
}

void internal_failure() {
  Harness h;
  h.config.transport_idle_ms = 5000;
  h.engine.windows = {{simple_page(1), true, 3}, {simple_page(1)}};
  h.start();
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 3s);
  auto call = h.stub->ParseDocument(&context);
  auto options = options_request(2);
  options.mutable_options()->set_abort_on_error(true);
  require(call->Write(options) && call->Write(page_request(1)), "partial input sent without half-close");
  v1::ParseDocumentResponse event;
  bool typed_failure = false;
  while (call->Read(&event)) {
    typed_failure |= event.has_page_failure() && event.page_failure().code() == v1::PAGE_FAILURE_CODE_INFERENCE;
  }
  const auto status = call->Finish();
  require(status.error_code() == grpc::StatusCode::CANCELLED || status.error_code() == grpc::StatusCode::ABORTED,
          "failure returns ABORTED or CANCELLED when a pending Read must be interrupted");
  std::println("internal failure wire status={}", static_cast<int>(status.error_code()));
  require(typed_failure, "readable client receives typed engine failure before cancellation");
  h.recovery();
}

void slow_reader() {
  Harness h;
  // Inference can outlast the transport release budget, especially with
  // sanitizers. Make that distinction deterministic instead of depending
  // on how quickly this machine tokenizes the large fake output below.
  h.engine.prefill_delay = 2500ms;
  std::string text = "<PAGE>";
  for (int i = 0; i < 40; ++i) text += "<BLOCK>text[1,2,3,4]<FORK>" + std::string(128 * 1024, 'x');
  h.engine.token_run = 128 * 1024;
  h.config.max_output_buffered_bytes = 64 * 1024 * 1024;
  h.engine.windows = {{text}, {simple_page(1)}};
  h.start();
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 30s);
  auto call = h.stub->ParseDocument(&context);
  require(call->Write(options_request(1)) && call->Write(page_request(1)), "request sent");
  const auto until = std::chrono::steady_clock::now() + 15s;
  while ((!h.engine.decodes || h.engine.concurrent) && std::chrono::steady_clock::now() < until) {
    std::this_thread::sleep_for(5ms);
  }
  const bool producer_finished = h.engine.decodes == 1 && h.engine.concurrent == 0 && h.engine.steps_emitted > 0;
  // Start the two-second release check after inference has returned, not
  // before output even exists. The client still has not read or cancelled.
  const bool recovered_without_reading = h.released();
  v1::ParseDocumentResponse event;
  while (call->Read(&event)) {}
  const auto status = call->Finish();
  require(producer_finished, "fake inference produced output and returned before transport release check");
  require(recovered_without_reading, "stalled Write releases budgets without client intervention");
  require(status.error_code() == grpc::StatusCode::CANCELLED, "blocked Write is cancelled on wire");
  h.engine.prefill_delay = 0ms;
  h.recovery();
}

void long_decode() {
  Harness h;
  h.engine.prefill_delay = 300ms;  // Twice the input-idle limit, with no output progress.
  h.engine.step_delay = 5ms;
  h.engine.windows = {{simple_page(1)}, {simple_page(2)}};
  h.start();
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 5s);
  auto call = h.stub->ParseDocument(&context);
  require(call->Write(options_request(2)) && call->Write(page_request(1)), "first window sent");
  v1::ParseDocumentResponse event;
  bool sent_second = false;
  while (call->Read(&event)) {
    if (event.has_page_document() && !sent_second) {
      require(call->Write(page_request(2)), "next window sent after long decode");
      sent_second = true;
    }
  }
  require(call->Finish().ok() && sent_second, "long inference does not consume input inactivity timeout");
  require(h.released(), "long inference releases budgets");
}

void resource_failure() {
  Harness h;
  h.engine.windows = {{simple_page(1)}};
  h.start();
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 3s);
  auto call = h.stub->ParseDocument(&context);
  require(call->Write(options_request(1)) && call->Write(page_request(1, "PNG 100000 100000")), "oversized raster sent");
  v1::ParseDocumentResponse event;
  while (call->Read(&event)) {}
  const auto status = call->Finish();
  require(status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED && !status.error_message().empty(),
          "server limit retains RESOURCE_EXHAUSTED when forced cancellation is unnecessary");
  h.recovery();
}

void queued_cancel() {
  Harness h;
  h.config.documents_in_flight = 2;
  h.engine.prefill_delay = 500ms;
  h.engine.windows = {{simple_page(1)}, {simple_page(1)}};
  h.start();
  grpc::ClientContext first, second;
  first.set_deadline(std::chrono::system_clock::now() + 3s);
  second.set_deadline(std::chrono::system_clock::now() + 3s);
  auto a = h.stub->ParseDocument(&first);
  require(a->Write(options_request(1)) && a->Write(page_request(1)), "first request sent");
  auto until = std::chrono::steady_clock::now() + 1s;
  while (!h.engine.decodes && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
  auto b = h.stub->ParseDocument(&second);
  require(b->Write(options_request(1)) && b->Write(page_request(1)), "queued request sent");
  until = std::chrono::steady_clock::now() + 100ms;
  while (h.service->document_slots().in_use() != 2 && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
  second.TryCancel();
  until = std::chrono::steady_clock::now() + 150ms;
  while (h.service->document_slots().in_use() != 1 && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(2ms);
  const bool released_while_first_busy = h.service->document_slots().in_use() == 1 && h.engine.concurrent == 1;
  v1::ParseDocumentResponse event;
  while (b->Read(&event)) {}
  const auto status = b->Finish();
  while (a->Read(&event)) {}
  const auto first_status = a->Finish();
  require(released_while_first_busy, "cancelled engine waiter releases admission before another decode finishes");
  require(status.error_code() == grpc::StatusCode::CANCELLED && first_status.ok(), "first decode survives cancellation of waiter");
  require(h.engine.decodes == 1 && h.released(), "cancelled waiter never enters engine");
}

int main(int argc, char** argv) {
  return run_tests([&] {
    require(argc == 2, "case required");
    const std::string name = argv[1];
    if (name == "idle_initial") idle(false);
    else if (name == "idle_pages") idle(true);
    else if (name == "admission") admission();
    else if (name == "failure") internal_failure();
    else if (name == "slow_reader") slow_reader();
    else if (name == "long_decode") long_decode();
    else if (name == "resource") resource_failure();
    else if (name == "queued_cancel") queued_cancel();
    else if (name == "no_halfclose") { Harness h; h.engine.windows = {{simple_page(1)}}; h.start(); h.recovery(); }
    else throw std::runtime_error("unknown case");
  });
}
