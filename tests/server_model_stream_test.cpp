#include <grpcpp/grpcpp.h>
// Real weights through the gRPC transport, with one and two pages, a
// slow reader and cancellation. Never silently substitutes a fake engine.
#include <chrono>
#include <thread>
#include "check.h"
#include "fixture_content.h"
#include "image/page_preprocess.h"
#include "model_document_checks.h"
#include "backend/reference/parallel.h"
#include "output_helpers.h"
#include "service/model_engine.h"
#include "service_helpers.h"

using namespace uocr;
using namespace uocr::testing;
using namespace std::chrono_literals;

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--check-fixture-assertions") return run_tests(check_fixture_assertions);
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
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    const auto server = builder.BuildAndStart();
    require(server != nullptr && port != 0, "real loopback server started");
    const auto stub = v1::UnlimitedOcrService::NewStub(grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
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
    const auto image = decode_image({reinterpret_cast<const std::uint8_t*>(png.data()), png.size()});
    std::map<int, std::vector<std::string>> document_snapshots;
    const auto verify_rpc = [&](int pages) {
      const auto request_start = std::chrono::steady_clock::now();
      std::println("starting {} page(s): backend={}, precision={}", pages, info.backend(), info.weight_format());
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + 20min);
      auto stream = stub->ParseDocument(&context);
      require(stream->Write(options_request(static_cast<std::uint32_t>(pages))), "options sent");
      for (int page = 1; page <= pages; ++page) {
        require(stream->Write(page_request(static_cast<std::uint32_t>(page), png)), "raster sent");
      }
      // Leave input open: all declared pages are sent, so long native
      // inference must not depend on client half-close or an idle timer.
      v1::ParseDocumentResponse event;
      int token_pages = 0;
      int fragments = 0;
      bool complete = false;
      std::vector<output::PageGeometry> geometry;
      for (int page = 1; page <= pages; ++page)
        geometry.push_back({static_cast<unsigned>(page), static_cast<unsigned>(image.width),
                            static_cast<unsigned>(image.height)});
      model_docs::StreamChecks document_checks(geometry);
      std::vector<v1::Block> all_blocks;
      std::vector<std::vector<v1::Block>> blocks(static_cast<std::size_t>(pages + 1));
      while (stream->Read(&event)) {
        document_checks.observe(event);
        if (event.has_page_failure()) throw std::runtime_error(event.page_failure().detail());
        if (event.has_tokens()) {
          ++token_pages;
          require(event.tokens().steps_size() > 0, "real decode produced tokens");
          std::println("raw page={}: {}", event.tokens().page_no(), event.tokens().text());
        }
        if (event.has_block()) {
          const auto& block = event.block();
          require(block.page_no() >= 1 && block.page_no() <= static_cast<unsigned>(pages), "block page is in request");
          blocks[block.page_no()].push_back(block);
          all_blocks.push_back(block);
          const auto& box = block.box();
          std::println("page={} kind={} box=[{},{},{},{}] text={}", block.page_no(), block.label_text(),
                       box.x1(), box.y1(), box.x2(), box.y2(), block.has_text() ? block.text().text() : "");
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
      for (int page = 1; page <= pages; ++page) {
        check_fixture_content(blocks[static_cast<std::size_t>(page)], static_cast<std::uint32_t>(page));
      }
      const auto documents = document_checks.finish(all_blocks);
      const auto [previous, inserted] = document_snapshots.emplace(pages, documents);
      require(inserted || previous->second == documents, "typed documents repeat exactly, including after cancellation");
      std::println("hardware RPC backend={} pages={} seconds={:.6f}: content, boxes, tokens and typed documents PASS",
                   engine->info().backend, pages,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - request_start).count());
    };
    const bool cancel_only = argc == 2 && std::string(argv[1]) == "--cancel-only";
    if (!cancel_only) {
      for (const int pages : {1, 2, 1, 2, 1, 2}) verify_rpc(pages);
    }
    for (int cycle = 0; cycle < 3; ++cycle) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 20min);
    auto stream = stub->ParseDocument(&context);
    require(stream->Write(options_request(1)) && stream->Write(page_request(1, png)), "cancel request sent");
    // Cancellation must also work without a client half-close.
    v1::ParseDocumentResponse event;
    require(stream->Read(&event) && event.has_window_started(), "real prefill completed");
    context.TryCancel();
    while (stream->Read(&event)) {}
    require(stream->Finish().error_code() == grpc::StatusCode::CANCELLED, "real-model stream cancelled");
    const auto released_by = std::chrono::steady_clock::now() + 30s;
    while (service.document_slots().in_use() != 0 && std::chrono::steady_clock::now() < released_by)
      std::this_thread::sleep_for(10ms);
    require(service.document_slots().in_use() == 0, "cancel releases admission before recovery");
    verify_rpc(1);
    std::println("real-model TCP cancellation/recovery cycle {} PASS", cycle);
    }
    server->Shutdown(std::chrono::system_clock::now() + 10s);
    server->Wait();
    require(service.buffered_bytes().in_use() == 0 && service.document_slots().in_use() == 0,
            "real-model cancellation returned leases");
  });
}
