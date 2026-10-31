// The service against the scripted engine: event order, windows,
// back-pressure, caps, cancellation, abort_on_error and every page failure
// code, then one round trip through a real in-process gRPC server.

#include <grpcpp/grpcpp.h>

#include <map>
#include <set>
#include <thread>

#include "check.h"
#include "service_helpers.h"

using namespace uocr;
using namespace uocr::testing;

namespace {

std::vector<WindowScript> pages_script(int pages, int window) {
  std::vector<WindowScript> out;
  for (int first = 1; first <= pages; first += window) {
    WindowScript w;
    for (int p = first; p <= std::min(pages, first + window - 1); ++p) w.output += simple_page(p);
    out.push_back(w);
  }
  return out;
}

void ordering() {
  service::Config config;
  config.page_window = 2;
  FakeEngine engine;
  engine.windows = pages_script(5, 2);
  service::UnlimitedOcrServiceImpl service(config, engine);
  const Outcome out = run(service, document(options_request(5)));
  require(out.status.ok(), "ok: " + out.status.error_message());
  require(engine.window_sizes == std::vector<std::size_t>({2, 2, 1}), "windows of 2, 2 and 1 pages");
  require(out.events.front().has_window_started(), "a window starts first");
  require(out.events.back().has_complete() && out.events[out.events.size() - 2].has_document(),
          "the folded Document, then ParseComplete, last");
  std::uint32_t last_page = 0;
  bool tokens_seen = false;
  for (const auto& e : out.events) {
    if (e.has_block()) require(e.block().page_no() > last_page, "blocks belong to the open page");
    if (e.has_tokens()) {
      require(e.tokens().page_no() == last_page + 1, "pages close strictly in order");
      last_page = e.tokens().page_no();
      tokens_seen = true;
    }
    if (e.has_page_document()) require(tokens_seen && e.page_document().page_no() == last_page, "fragment after tokens");
  }
  require(out.blocks().size() == 10, "two blocks per page");
  const auto& complete = out.events.back().complete();
  require(complete.pages_ok() == 5 && complete.pages_failed() == 0, "five pages ok");
  require(complete.prompt_tokens() == 500, "prompt tokens over the windows");
  engine.windows.push_back(pages_script(1, 1)[0]);
  const auto quiet = run(service, document(options_request(1, false)));
  require(quiet.status.ok() && quiet.blocks().empty() && quiet.tokens().size() == 1,
          "blocks only when asked for, tokens always");
}

void caps_and_invalid_input() {
  service::Config config;
  config.max_pages = 3;
  config.max_page_bytes = 1024;
  FakeEngine engine;
  engine.windows = pages_script(3, 10);
  service::UnlimitedOcrServiceImpl service(config, engine);
  auto code = [&](std::vector<v1::ParseDocumentRequest> rs) { return run(service, std::move(rs)).status.error_code(); };
  require(code({page_request(1)}) == grpc::StatusCode::INVALID_ARGUMENT, "options must come first");
  require(code({options_request(0)}) == grpc::StatusCode::INVALID_ARGUMENT, "page_count is required");
  require(code({options_request(4)}) == grpc::StatusCode::RESOURCE_EXHAUSTED, "page cap");
  require(code({options_request(2), page_request(3)}) == grpc::StatusCode::INVALID_ARGUMENT, "page_no in range");
  require(code({options_request(2), page_request(1), page_request(1)}) == grpc::StatusCode::INVALID_ARGUMENT,
          "a page twice");
  require(code({options_request(2), page_request(1, "PNG 1 1" + std::string(2000, ' '))}) ==
              grpc::StatusCode::RESOURCE_EXHAUSTED,
          "page byte cap");
  require(code({options_request(2), page_request(1)}) == grpc::StatusCode::INVALID_ARGUMENT, "missing pages");
  require(code({options_request(1), options_request(1)}) == grpc::StatusCode::INVALID_ARGUMENT, "options twice");
}

void back_pressure() {
  service::Config config;
  config.page_window = 1;
  config.max_stream_buffered_bytes = 30;  // two 13-byte rasters
  FakeEngine engine;
  engine.windows = pages_script(5, 1);
  engine.step_delay = std::chrono::microseconds(200);
  service::UnlimitedOcrServiceImpl service(config, engine);
  require(run(service, document(options_request(5))).status.ok(), "in-order pages flow through a small budget");
  std::vector<v1::ParseDocumentRequest> reversed{options_request(5)};
  for (std::uint32_t p = 5; p >= 1; --p) reversed.push_back(page_request(p));
  engine.windows = pages_script(5, 1);
  engine.decodes = 0;
  const Outcome stuck = run(service, reversed);
  require(stuck.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
          "pages far ahead of the window fill the budget: " + stuck.status.error_message());
  require(service.buffered_bytes().in_use() == 0, "every byte is given back");
}

void cancellation() {
  service::Config config;
  FakeEngine engine;
  engine.windows = pages_script(3, 10);
  engine.step_delay = std::chrono::microseconds(500);
  service::UnlimitedOcrServiceImpl service(config, engine);
  const Outcome out = run(service, document(options_request(3)), 3);
  require(out.status.error_code() == grpc::StatusCode::CANCELLED, "a client that goes away cancels the stream");
  require(!out.events.empty() && !out.events.back().has_complete(), "no trailer after cancellation");
}

void failure_codes() {
  service::Config config;
  config.page_window = 1;
  config.max_tokens_per_page = 60;
  FakeEngine engine;
  engine.windows = std::vector<WindowScript>(4);
  engine.windows[0].output = "<PAGE><BLOCK>text[1,2,3,4]<FORK>" + std::string(200, 'x');  // over the cap
  engine.windows[1].output = "<PAGE>";
  for (int i = 0; i < 10; ++i) engine.windows[1].output += "no blocks at all ";
  engine.windows[1].ban_from = 0;
  engine.windows[2].output = simple_page(4);
  engine.windows[2].throw_after = 3;
  engine.windows[3].output = simple_page(5);
  service::UnlimitedOcrServiceImpl service(config, engine);
  std::vector<v1::ParseDocumentRequest> rs = document(options_request(5));
  rs[3] = page_request(3, "GIF89a");
  const Outcome out = run(service, rs);
  require(out.status.ok(), "failures do not end the stream without abort_on_error");
  std::map<std::uint32_t, std::set<v1::PageFailureCode>> codes;
  for (const auto& f : out.failures()) codes[f.page_no()].insert(f.code());
  require(codes[1].contains(v1::PAGE_FAILURE_CODE_TOKEN_CAP), "token cap");
  require(codes[2].contains(v1::PAGE_FAILURE_CODE_UNPARSEABLE_OUTPUT), "unparseable output");
  require(codes[2].contains(v1::PAGE_FAILURE_CODE_REPETITION), "repetition");
  require(codes[3].contains(v1::PAGE_FAILURE_CODE_IMAGE_DECODE), "image decode");
  require(codes[4].contains(v1::PAGE_FAILURE_CODE_INFERENCE), "inference");
  require(!codes.contains(5), "page 5 is fine");
  require(out.tokens().size() == 5, "every page still sends its tokens");
  require(out.events.back().complete().pages_failed() == 4, "four failed pages");

  engine.windows = std::vector<WindowScript>(2);
  engine.windows[0].output = simple_page(1);
  engine.windows[1].output = simple_page(3);
  engine.decodes = 0;
  auto abort = document(options_request(3));
  abort[0].mutable_options()->set_abort_on_error(true);
  abort[2] = page_request(2, "not an image");
  const Outcome aborted = run(service, abort);
  require(aborted.status.error_code() == grpc::StatusCode::ABORTED, "abort_on_error ends the stream");
  require(aborted.failures().size() == 1 && aborted.failures()[0].page_no() == 2, "after sending the failure");
  require(engine.decodes == 1, "nothing after the failed page is decoded");
}

void one_document_at_a_time() {
  service::Config config;
  FakeEngine engine;
  engine.windows = pages_script(1, 1);
  const auto more = pages_script(1, 1);
  engine.windows.insert(engine.windows.end(), {more[0], more[0], more[0]});
  engine.step_delay = std::chrono::microseconds(300);
  service::UnlimitedOcrServiceImpl service(config, engine);
  std::vector<std::thread> clients;
  std::atomic<int> ok{0};
  for (int i = 0; i < 4; ++i) {
    clients.emplace_back([&] { ok += run(service, document(options_request(1))).status.ok() ? 1 : 0; });
  }
  for (auto& c : clients) c.join();
  require(ok == 4, "four streams complete");
  require(engine.max_concurrent == 1, "one document decodes at a time");
}

void over_grpc() {
  service::Config config;
  FakeEngine engine;
  engine.windows = pages_script(2, 10);
  service::UnlimitedOcrServiceImpl service(config, engine);
  grpc::ServerBuilder builder;
  builder.RegisterService(&service);
  const auto server = builder.BuildAndStart();
  const auto stub = v1::UnlimitedOcrService::NewStub(server->InProcessChannel({}));
  grpc::ClientContext info_context;
  v1::GetServiceInfoResponse info;
  require(stub->GetServiceInfo(&info_context, {}, &info).ok(), "GetServiceInfo");
  require(info.backend() == "fake" && info.ui().path() == "/ui/unlimited-ocr" && info.page_window() == 10,
          "service info");
  grpc::ClientContext context;
  auto stream = stub->ParseDocument(&context);
  for (const auto& r : document(options_request(2))) require(stream->Write(r), "write");
  stream->WritesDone();
  v1::ParseDocumentResponse e;
  int tokens = 0;
  bool complete = false;
  while (stream->Read(&e)) {
    tokens += e.has_tokens() ? 1 : 0;
    complete = e.has_complete();
  }
  require(stream->Finish().ok() && tokens == 2 && complete, "a document over gRPC");
  server->Shutdown();
}

}  // namespace

int main() {
  return run_tests([] {
    ordering();
    caps_and_invalid_input();
    back_pressure();
    cancellation();
    failure_codes();
    one_document_at_a_time();
    over_grpc();
  });
}
