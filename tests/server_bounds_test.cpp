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

int main(int argc, char** argv) {
  return run_tests([&] {
    require(argc == 2, "case required");
    const std::string test = argv[1];
    if (test == "duplicate") consumed_duplicate();
    else if (test == "global") stalled_window(true);
    else if (test == "stream") stalled_window(false);
    else if (test == "pixels") pixel_limit();
    else if (test == "output") output_bound();
    else throw std::runtime_error("unknown case");
  });
}
