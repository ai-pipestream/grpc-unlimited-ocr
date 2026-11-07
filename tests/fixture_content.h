#pragma once

#include <string>
#include <vector>
#include "check.h"
#include "output_helpers.h"

namespace uocr::testing {

// Check recognizable content at separated positions on the known raster,
// without requiring exact tokens, punctuation or block counts.
inline void check_fixture_content(const std::vector<v1::Block>& blocks, std::uint32_t page) {
  struct Anchor { const char* text; std::uint32_t top; std::uint32_t bottom; };
  const Anchor anchors[] = {
      {"Scanned page: OCR", 0, 200},
      {"no digital text layer", 50, 300},
      {"quarterly volume", 400, 650},
      {"QR payload", 700, 1000}};
  require(!blocks.empty(), "fixture has layout blocks");
  for (const auto& block : blocks) {
    require(block.page_no() == page, "fixture block belongs to requested page");
    require(block.has_box() && block.raw_box_size() == 4, "fixture block has four box coordinates");
    for (const auto coordinate : block.raw_box()) {
      require(coordinate >= 0 && coordinate <= 1000, "raw geometry stays inside the page without clamping");
    }
    require(block.raw_box(0) < block.raw_box(2) && block.raw_box(1) < block.raw_box(3),
            "raw geometry has positive area");
    const auto& box = block.box();
    require(box.x1() < box.x2() && box.y1() < box.y2() && box.x2() <= 1000 && box.y2() <= 1000,
            "normalized geometry has positive area inside the page");
  }
  for (const auto& anchor : anchors) {
    bool found = false;
    for (const auto& block : blocks) {
      if (block.has_text() && block.text().text().find(anchor.text) != std::string::npos) {
        require(block.box().y1() >= anchor.top && block.box().y2() <= anchor.bottom,
                std::string("fixture text has plausible vertical placement: ") + anchor.text);
        found = true;
      }
    }
    require(found, std::string("recognized fixture text: ") + anchor.text);
  }
}

inline void check_fixture_assertions() {
  auto blocks = parse_whole(read_text(std::string(UOCR_GOLDENS) + "/quickstart-p2/output_raw.txt").substr(6));
  check_fixture_content(blocks, 1);
  auto bad = blocks;
  bad.front().mutable_text()->set_text("unrelated text");
  require_throws([&] { check_fixture_content(bad, 1); }, "nonempty unrelated text must fail");
  bad = blocks;
  bad.front().set_raw_box(0, -1);
  require_throws([&] { check_fixture_content(bad, 1); }, "clamped raw coordinates must fail");
  bad = blocks;
  bad.front().mutable_box()->set_x2(bad.front().box().x1());
  require_throws([&] { check_fixture_content(bad, 1); }, "zero-area box must fail");
  bad = blocks;
  bad.front().mutable_box()->set_y1(700);
  bad.front().mutable_box()->set_y2(800);
  require_throws([&] { check_fixture_content(bad, 1); }, "valid box at wrong position must fail");
  require_throws([&] { check_fixture_content(blocks, 2); }, "wrong page must fail");
}

}  // namespace uocr::testing
