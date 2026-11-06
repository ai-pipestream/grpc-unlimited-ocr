#include <grpcpp/grpcpp.h>
#include <future>
#include <thread>
#include "check.h"
#include "service_helpers.h"
#include "service/channel.h"

using namespace uocr;
using namespace uocr::testing;
using namespace std::chrono_literals;

void consumed_duplicate() {
  service::PageIntake intake(2, 100);
  service::IntakePage page;
  page.page_no = 1;
  page.raster = "x";
  intake.add(std::move(page));
  auto taken = intake.take(1, 1);
  require(taken.has_value(), "page consumed");
  require(intake.has(1), "consumed page must remain in duplicate history");
}

void stalled_window(bool global) {
  service::Config config;
  config.page_window = 2;
  config.max_stream_buffered_bytes = global ? 100 : 15;
  config.max_buffered_bytes = global ? 15 : 100;
  FakeEngine engine;
  engine.windows = {{simple_page(1) + simple_page(2)}};
  service::UnlimitedOcrServiceImpl service(config, engine);
  std::atomic<bool> cancelled{false};
  std::size_t i = 0;
  auto requests = document(options_request(2));
  auto task = std::async(std::launch::async, [&] {
    return service.ParseDocumentCore(
        [&](v1::ParseDocumentRequest* r) { if (i == requests.size()) return false; *r = requests[i++]; return true; },
        [](const auto&) { return true; }, [&] { return cancelled.load(); });
  });
  const bool ready = task.wait_for(1s) == std::future_status::ready;
  cancelled = true;
  const auto status = task.get();
  require(ready, "oversized window must return without waiting for cancellation");
  require(status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "oversized window must respect its budget");
  require(service.buffered_bytes().in_use() == 0, "leases returned");
}

void pixel_limit() {
  service::Config config;
  FakeEngine engine;
  engine.windows = {{simple_page(1)}};
  service::UnlimitedOcrServiceImpl service(config, engine);
  auto page = page_request(1, "PNG 100000 100000");
  page.mutable_page()->set_width_px(1);
  page.mutable_page()->set_height_px(1);
  const auto out = run(service, {options_request(1), page});
  require(out.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
          "actual raster dimensions must be capped even with forged metadata");
  require(engine.decodes == 0, "oversized raster never decoded");
}

void output_bound() {
  service::Channel<v1::ParseDocumentResponse> channel;
  std::atomic<int> pushed{0};
  auto task = std::async(std::launch::async, [&] {
    for (int i = 0; i < 64; ++i) {
      v1::ParseDocumentResponse event;
      event.mutable_tokens()->set_text(std::string(1024 * 1024, 'x'));
      channel.push(std::move(event));
      ++pushed;
    }
  });
  const bool blocked = task.wait_for(300ms) == std::future_status::timeout;
  channel.close();
  const bool woke = task.wait_for(1s) == std::future_status::ready;
  task.get();
  require(blocked, "slow reader must block output producer within its byte cap");
  require(woke, "closing output must wake a blocked producer");
}

void oversized_response() {
  service::Config config;
  config.max_output_buffered_bytes = 1024;
  FakeEngine engine;
  engine.windows = {{"<PAGE><BLOCK>text[1,2,3,4]<FORK>" + std::string(2000, 'x')}};
  service::UnlimitedOcrServiceImpl service(config, engine);
  const auto out = run(service, document(options_request(1)));
  require(out.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "oversized response returns an error");
  require(service.buffered_bytes().in_use() == 0 && service.document_slots().in_use() == 0,
          "output failure releases input and document leases");
}

void grpc_cancellation() {
  service::Config config;
  FakeEngine engine;
  engine.windows = {{simple_page(1)}};
  engine.step_delay = 2ms;
  service::UnlimitedOcrServiceImpl service(config, engine);
  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  const auto server = builder.BuildAndStart();
  const auto stub = v1::UnlimitedOcrService::NewStub(server->InProcessChannel({}));
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 5s);
  auto stream = stub->ParseDocument(&context);
  for (const auto& request : document(options_request(1))) require(stream->Write(request), "request written");
  // Leave the input open: cancellation must also release the blocked RPC Read.
  v1::ParseDocumentResponse event;
  require(stream->Read(&event) && event.has_window_started(), "decode began");
  context.TryCancel();
  while (stream->Read(&event)) {}
  require(stream->Finish().error_code() == grpc::StatusCode::CANCELLED, "transport cancellation");
  server->Shutdown(std::chrono::system_clock::now() + 2s);
  server->Wait();
  require(service.buffered_bytes().in_use() == 0 && service.document_slots().in_use() == 0,
          "cancelled RPC releases leases");
}

void global_retry() {
  service::Config config;
  config.page_window = 1;
  config.max_buffered_bytes = 15;
  FakeEngine engine;
  engine.windows = {{simple_page(1)}, {simple_page(1)}};
  engine.step_delay = 2ms;
  service::UnlimitedOcrServiceImpl service(config, engine);
  auto first = std::async(std::launch::async, [&] { return run(service, document(options_request(1))); });
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (engine.decodes == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  require(engine.decodes == 1, "first stream holds its raster while decoding");
  const auto refused = run(service, document(options_request(1)));
  require(refused.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "competing stream fails promptly");
  require(first.get().status.ok(), "first stream unaffected");
  require(run(service, document(options_request(1))).status.ok(), "retry succeeds after leases are released");
  require(service.buffered_bytes().in_use() == 0 && service.document_slots().in_use() == 0, "all leases released");
}

void slow_reader_cancel() {
  service::Config config;
  config.max_output_buffered_bytes = 1024;
  FakeEngine engine;
  std::string raw = "<PAGE>";
  for (int i = 0; i < 1000; ++i) raw += "<BLOCK>text[1,2,3,4]<FORK>some text";
  engine.windows = {{raw}};
  service::UnlimitedOcrServiceImpl service(config, engine);
  std::atomic<bool> cancelled{false};
  std::promise<void> writer_entered;
  auto entered = writer_entered.get_future();
  auto requests = document(options_request(1));
  std::size_t next = 0;
  auto task = std::async(std::launch::async, [&] {
    return service.ParseDocumentCore(
        [&](v1::ParseDocumentRequest* r) { if (next == requests.size()) return false; *r = requests[next++]; return true; },
        [&](const auto&) {
          writer_entered.set_value();
          while (!cancelled) std::this_thread::sleep_for(1ms);
          return false;
        }, [&] { return cancelled.load(); });
  });
  const bool writing = entered.wait_for(2s) == std::future_status::ready;
  std::this_thread::sleep_for(100ms);
  const auto before = engine.steps_emitted.load();
  std::this_thread::sleep_for(100ms);
  const bool blocked = engine.concurrent == 1 && before > 0 && before == engine.steps_emitted.load();
  cancelled = true;
  const auto result = task.get();
  require(writing && blocked, "slow reader applies back-pressure to model decode");
  require(result.error_code() == grpc::StatusCode::CANCELLED, "cancel wakes the blocked output producer");
  require(service.buffered_bytes().in_use() == 0 && service.document_slots().in_use() == 0, "leases returned");
}

void slow_reader_slots() {
  service::Config config;
  FakeEngine engine;
  engine.windows = {{simple_page(1)}, {simple_page(1)}};
  service::UnlimitedOcrServiceImpl service(config, engine);
  std::atomic<bool> release_writer{false};
  auto requests = document(options_request(1));
  std::size_t next = 0;
  auto first = std::async(std::launch::async, [&] {
    return service.ParseDocumentCore(
        [&](v1::ParseDocumentRequest* r) { if (next == requests.size()) return false; *r = requests[next++]; return true; },
        [&](const auto&) {
          while (!release_writer) std::this_thread::sleep_for(1ms);
          return true;
        }, [] { return false; });
  });
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while ((engine.decodes == 0 || engine.concurrent != 0) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const bool decoded = engine.decodes == 1 && engine.concurrent == 0;
  auto second = std::async(std::launch::async, [&] { return run(service, document(options_request(1))); });
  const bool blocked = second.wait_for(200ms) == std::future_status::timeout;
  const bool slot_held = service.document_slots().in_use() == 1 && engine.decodes == 1;
  release_writer = true;
  const auto first_status = first.get();
  const auto second_out = second.get();
  require(decoded && blocked && slot_held, "completed decode retains its slot while responses wait for a reader");
  require(first_status.ok() && second_out.status.ok(), "both streams finish after reader resumes");
  require(service.document_slots().in_use() == 0, "slots released after RPC completion");
}

int main(int argc, char** argv) {
  return run_tests([&] {
    require(argc == 2, "case required");
    const std::string test = argv[1];
    if (test == "duplicate") consumed_duplicate();
    else if (test == "global") stalled_window(true);
    else if (test == "stream") stalled_window(false);
    else if (test == "pixels") pixel_limit();
    else if (test == "output") output_bound();
    else if (test == "oversized_response") oversized_response();
    else if (test == "grpc_cancellation") grpc_cancellation();
    else if (test == "global_retry") global_retry();
    else if (test == "slow_reader_cancel") slow_reader_cancel();
    else if (test == "slow_reader_slots") slow_reader_slots();
    else throw std::runtime_error("unknown case");
  });
}
