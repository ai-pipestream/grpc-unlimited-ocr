// Real native Intel model through typed gRPC, including repeated pages,
// in-flight cancellation and content-checked recovery. No fake engine.
#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <thread>

#include "check.h"
#include "fixture_content.h"
#include "output_helpers.h"
#include "service/model_engine.h"
#include "service_helpers.h"
using namespace uocr;
using namespace uocr::testing;
using namespace std::chrono_literals;
namespace {
std::uint64_t anonymous_bytes() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line))
    if (line.starts_with("RssAnon:")) return std::stoull(line.substr(8)) * 1024;
  throw std::runtime_error("cannot read process anonymous memory");
}
void idle(service::UnlimitedOcrServiceImpl& svc) {
  const auto until = std::chrono::steady_clock::now() + 10s;
  while ((svc.buffered_bytes().in_use() != 0 || svc.document_slots().in_use() != 0) &&
         std::chrono::steady_clock::now() < until)
    std::this_thread::sleep_for(10ms);
  require(svc.buffered_bytes().in_use() == 0 && svc.document_slots().in_use() == 0,
          "all input and document leases returned");
}
std::string parse(v1::UnlimitedOcrService::Stub& stub, service::UnlimitedOcrServiceImpl& svc, const std::string& png,
                  int pages) {
  const auto start = std::chrono::steady_clock::now();
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 10min);
  auto stream = stub.ParseDocument(&context);
  require(stream->Write(options_request(static_cast<std::uint32_t>(pages))), "options sent");
  for (int p = 1; p <= pages; ++p)
    require(stream->Write(page_request(static_cast<std::uint32_t>(p), png)), "page sent");
  v1::ParseDocumentResponse event;
  int fragments = 0, token_pages = 0;
  bool complete = false;
  std::string raw;
  std::vector<std::vector<v1::Block>> blocks(static_cast<std::size_t>(pages + 1));
  while (stream->Read(&event)) {
    require(!event.has_page_failure(), "no page failure: " + event.page_failure().detail());
    if (event.has_tokens()) {
      ++token_pages;
      require(event.tokens().steps_size() > 0, "nonempty typed token trace");
      raw += event.tokens().text();
      for (const auto& t : event.tokens().steps())
        require(std::isfinite(t.logprob()) && std::isfinite(t.model_logprob()), "finite typed log probabilities");
      std::println("page={} raw={}", event.tokens().page_no(), event.tokens().text());
    }
    if (event.has_block()) {
      const auto& b = event.block();
      require(b.page_no() >= 1 && b.page_no() <= static_cast<unsigned>(pages), "block page range");
      blocks[b.page_no()].push_back(b);
    }
    if (event.has_page_document()) ++fragments;
    if (event.has_complete()) {
      complete = true;
      require(event.complete().pages_ok() == static_cast<unsigned>(pages) && event.complete().pages_failed() == 0,
              "all pages successful");
    }
    std::this_thread::sleep_for(10ms);
  }
  require(stream->Finish().ok() && complete && fragments == pages && token_pages == pages,
          "complete typed stream without client half-close");
  for (int p = 1; p <= pages; ++p)
    check_fixture_content(blocks[static_cast<std::size_t>(p)], static_cast<std::uint32_t>(p));
  idle(svc);
  std::println("RPC pages={} total_s={:.6f} rss_anon_bytes={}", pages,
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), anonymous_bytes());
  return raw;
}
void cancel(v1::UnlimitedOcrService::Stub& stub, service::UnlimitedOcrServiceImpl& svc, const std::string& png) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 10min);
  auto stream = stub.ParseDocument(&context);
  require(stream->Write(options_request(1)) && stream->Write(page_request(1, png)), "cancel request sent");
  v1::ParseDocumentResponse event;
  require(stream->Read(&event) && event.has_window_started(), "real GPU prefill completed before cancellation");
  const auto start = std::chrono::steady_clock::now();
  context.TryCancel();
  while (stream->Read(&event)) {
  }
  require(stream->Finish().error_code() == grpc::StatusCode::CANCELLED, "in-flight native decode cancelled");
  idle(svc);
  std::println("CANCEL after_prefill_ms={:.3f}",
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}
}  // namespace
int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (!models) return skip("UOCR_MODELS_DIR absent");
  return run_tests([&] {
    service::Config config;
    config.backend = "intel-gpu";
    config.weight_format = "bf16";
    config.models_dir = models;
    config.device_memory_cap = 12ULL << 30;
    config.page_window = 2;
    config.max_pages = 2;
    config.max_tokens_per_page = 600;
    const auto engine = service::load_model_engine(config);
    service::UnlimitedOcrServiceImpl svc(config, *engine);
    grpc::ServerBuilder builder;
    builder.RegisterService(&svc);
    const auto server = builder.BuildAndStart();
    require(server != nullptr, "server started");
    struct Shutdown {
      grpc::Server& server;
      ~Shutdown() {
        server.Shutdown(std::chrono::system_clock::now() + 10s);
        server.Wait();
      }
    } shutdown{*server};
    const auto stub = v1::UnlimitedOcrService::NewStub(server->InProcessChannel({}));
    grpc::ClientContext info_ctx;
    v1::GetServiceInfoResponse info;
    require(stub->GetServiceInfo(&info_ctx, {}, &info).ok(), "typed service info");
    require(info.backend() == "intel-gpu" && info.weight_format() == "bf16" &&
                info.device().find("Intel") != std::string::npos,
            "actual Intel backend identity");
    std::println("RPC backend={} device={} precision={} manifest={}", info.backend(), info.device(),
                 info.weight_format(), info.manifest_sha256());
    const auto png = read_text(std::string(UOCR_GOLDENS) + "/quickstart-p2/page.png");
    const auto one = parse(*stub, svc, png, 1), two = parse(*stub, svc, png, 2);
    const auto warm = anonymous_bytes();
    for (int repeat = 0; repeat < 3; ++repeat) {
      cancel(*stub, svc, png);
      require(parse(*stub, svc, png, 1) == one, "one-page recovery deterministic and content checked");
      require(parse(*stub, svc, png, 2) == two, "two-page repeat deterministic and content checked");
      require(anonymous_bytes() <= warm + (128ULL << 20), "anonymous RSS bounded after shape warmup");
    }
  });
}
