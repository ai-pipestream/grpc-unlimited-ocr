#pragma once

// The incremental block parser for one page: decode steps go in, typed
// Blocks come out the moment each one closes (when the next block starts,
// or when the page closes). The blocks are the same whether the steps
// arrive one at a time or all at once (see block_grammar.h).

#include <cstdint>
#include <string_view>
#include <vector>

#include "ai/pipestream/unlimitedocr/v1/unlimited_ocr.pb.h"
#include "output/block_grammar.h"
#include "output/page_text.h"

namespace uocr::output {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

// The kind and continuation a label names: "cross_table" is a TABLE that
// continues on the next page. Unknown labels are UNRECOGNISED.
v1::BlockKind block_kind(std::string_view label);
v1::Continuation block_continuation(std::string_view label);

class BlockParser {
 public:
  explicit BlockParser(std::uint32_t page_no) : page_no_(page_no) {}

  // One decode step's decoded bytes (never the <PAGE> marker or EOS
  // step). Returns the blocks this step closed.
  std::vector<v1::Block> add(std::uint32_t step, std::string_view bytes);

  // Closes the page: returns the blocks still open.
  std::vector<v1::Block> finish();

  const PageText& text() const { return text_; }

  // The page had text but the block grammar placed none of it.
  bool unparseable() const { return unparseable_; }

 private:
  std::vector<v1::Block> take(std::size_t count);

  std::uint32_t page_no_;
  PageText text_;
  std::vector<BlockSpan> spans_;
  std::size_t emitted_ = 0;
  bool unparseable_ = false;
};

// One span of a page's text as a Block (exposed for tests).
v1::Block make_block(std::uint32_t page_no, std::uint32_t ordinal, const BlockSpan& span, const PageText& text);

}  // namespace uocr::output
