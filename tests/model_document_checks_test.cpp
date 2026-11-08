#include "model_document_checks.h"

#include "output/document_fold.h"
#include "output_helpers.h"

using namespace uocr;
using namespace uocr::testing;

int main() {
  return run_tests([] {
    const std::vector<output::PageGeometry> pages{{1, 800, 1000}, {2, 800, 1000}};
    std::vector<v1::Block> blocks;
    std::vector<v1::ParseDocumentResponse> events;
    for (const auto& page : pages) {
      auto page_blocks = parse_whole(read_text(std::string(UOCR_GOLDENS) + "/quickstart-p2/output_raw.txt").substr(6));
      for (auto& block : page_blocks) block.set_page_no(page.page_no);
      blocks.insert(blocks.end(), page_blocks.begin(), page_blocks.end());
      events.emplace_back().mutable_tokens()->set_page_no(page.page_no);
      auto* fragment = events.emplace_back().mutable_page_document();
      fragment->set_page_no(page.page_no);
      output::DocumentInput input;
      input.pages = {page};
      *fragment->mutable_document() = output::page_document(input, page_blocks);
    }
    output::DocumentInput input;
    input.pages = pages;
    *events.emplace_back().mutable_document() = output::fold_document(input, blocks, {});
    const auto check = [&](const auto& candidate) {
      model_docs::StreamChecks checker(pages);
      for (const auto& event : candidate) checker.observe(event);
      return checker.finish(blocks);
    };
    require(check(events) == check(events), "valid typed fixture and deterministic snapshot");
    auto bad = events;
    bad[1].mutable_page_document()->mutable_document()->Clear();
    require_throws([&] { check(bad); }, "empty fragment rejected");
    bad = events;
    *bad[1].mutable_page_document()->mutable_document() = events[3].page_document().document();
    require_throws([&] { check(bad); }, "swapped fragment rejected");
    bad = events;
    bad[3] = bad[1];
    require_throws([&] { check(bad); }, "duplicate fragment rejected");
    bad = events;
    bad.pop_back();
    require_throws([&] { check(bad); }, "missing folded document rejected");
    bad = events;
    bad.back().mutable_document()->mutable_body()->mutable_children(0)->set_ref("#/texts/999");
    require_throws([&] { check(bad); }, "invalid folded reference rejected");
    bad = events;
    bad.back().mutable_document()->mutable_pictures(0)->mutable_captions(0)->set_ref("#/texts/999");
    require_throws([&] { check(bad); }, "invalid caption reference rejected");
    bad = events;
    bad[2].mutable_tokens()->set_page_no(1);
    require_throws([&] { check(bad); }, "duplicate token page rejected");
    bad = events;
    bad.back().mutable_document()->mutable_texts(0)->mutable_title()->mutable_base()->set_text("corrupt");
    require_throws([&] { check(bad); }, "corrupt folded text rejected");
    bad = events;
    bad[1].mutable_page_document()->mutable_document()->mutable_pictures(0)->mutable_prov(0)->mutable_bbox()->set_r(0);
    require_throws([&] { check(bad); }, "corrupt fragment geometry rejected");
    bad = events;
    bad.push_back(events.back());
    require_throws([&] { check(bad); }, "duplicate folded document rejected");
    std::println("typed model document mutations: 10 rejected");
  });
}
