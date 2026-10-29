// Page bookkeeping on a hand-built two-page prompt.

#include "check.h"
#include "model/page_layout.h"
#include "runtime/prompt_builder.h"

using namespace uocr;

int main() {
  return run_tests([] {
    // BOS, <PAGE>, 3 images, <PAGE>, 2 images, 4 instruction tokens.
    const PromptTokens t;
    const auto ids = build_prompt({3, 2}, t);
    require(ids.size() == 1 + 1 + 3 + 1 + 2 + 4, "prompt length");
    require(image_positions(ids) == std::vector<std::int64_t>{2, 3, 4, 6, 7}, "image positions");

    const PromptLayout layout = layout_prompt(ids);
    require(layout.prompt_pages == 2 && layout.rope_pages == 3, "two pages plus the empty bucket");
    // BOS is bucket 0; page 1 = <PAGE> + 3 images; page 2 = <PAGE> + 2 images;
    // the instruction falls back to bucket 0 and keeps counting there.
    const std::vector<std::int64_t> pages{0, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0};
    const std::vector<std::int64_t> toks{0, 0, 1, 2, 3, 0, 1, 2, 1, 2, 3, 4};
    for (std::size_t i = 0; i < ids.size(); ++i) {
      require(layout.coords[i].page == pages[i] && layout.coords[i].tok == toks[i], "rope coordinate at " + std::to_string(i));
    }
    require(layout.prefill_count == std::vector<std::int64_t>{1, 4, 3}, "prefill positions per bucket");
    require(layout.image_page == std::vector<std::int64_t>{-1, -1, 0, 0, 0, -1, 1, 1, -1, -1, -1, -1}, "image pages");
    require(layout.query_page == std::vector<std::int64_t>{-1, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1}, "query pages");

    const BlockPages blocks = block_image_pages(layout, 4);
    require(blocks.min_page == std::vector<std::int64_t>{0, 0, -1} && blocks.max_page == std::vector<std::int64_t>{0, 1, -1},
            "image pages per 4-token block");

    DecodeState state(layout, ids);
    require(state.query_page() == -1, "no response page yet");
    RopeCoord c = state.advance(t.page);  // response page 1 starts after its 4 prefill positions
    require(c.page == 1 && c.tok == 4 && state.query_page() == 0, "first response marker");
    c = state.advance(42);
    require(c.page == 1 && c.tok == 5 && state.query_page() == 0, "text on page 1");
    c = state.advance(t.page);
    require(c.page == 2 && c.tok == 3 && state.query_page() == 1, "second response marker");
    c = state.advance(t.page);  // a third marker clamps to the last page
    require(c.page == 2 && c.tok == 3, "extra marker clamps");

    require_throws([] { layout_prompt({0, 5, 6}); }, "prompt without images");
  });
}
