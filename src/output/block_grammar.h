#pragma once

// The block grammar of the model's output, ported from upstream's
// utils/process_unlimitedOCR2.py (baidu/Unlimited-OCR, MIT):
//
//   <BLOCK>label [<LEVELn>] [x1,y1,x2,y2] [<LEVELn>] [<FORK>] content
//
// with level and box in either order, a bare "label[x1,y1,x2,y2]" at the
// start of a page, a line or after <FORK> repaired into a block when the
// label is a known one, and text before the first block kept as an
// unrecognised span. Pages are split by the caller on the <PAGE> token.
//
// split_blocks is a pure function of one page's text. Every span but the
// last is final once a later span starts: text appended to the page can
// change only the last span. That is what lets the incremental parser emit
// a block the moment the next one starts and still give the same blocks
// whether the text arrives one token at a time or all at once.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace uocr::output {

struct BlockSpan {
  std::size_t begin = 0;          // first byte of the span (the block's header)
  std::size_t content_begin = 0;  // first byte after the header
  std::size_t end = 0;            // one past the last byte (the next span's begin, or the page's end)
  bool recognised = false;        // false: text the grammar could not place, kept verbatim
  std::string label;              // as generated; empty for unrecognised spans
  bool has_level = false;         // a <LEVEL> tag was written
  std::optional<std::int32_t> level;  // its number, when it carried one
  std::vector<std::int32_t> raw_box;  // every coordinate as written; empty when no box
  bool merge_with_previous = false;   // a <MERGE> sits between the previous span's header and this one
};

std::vector<BlockSpan> split_blocks(std::string_view page);

// Content with <MERGE> and the other layout control tokens removed and
// outer whitespace trimmed, as upstream renders block text.
std::string clean_content(std::string_view content);

// Outer ASCII whitespace trimmed.
std::string_view trim(std::string_view text);

}  // namespace uocr::output
