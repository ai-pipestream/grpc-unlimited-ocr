#pragma once

// Which blocks continue which, decided as each block arrives, by upstream's
// grouping rules (process_unlimitedOCR2.py, _build_groups):
//
//  - text and table chains: a cross_ block (or any table) opens a chain for
//    its kind, and an append_ block of the same kind joins the open chain.
//    A table chain does not jump over a page that holds another table.
//  - a block written after <MERGE> joins the block before it when both are
//    on one page and both are text or title.
//
// Each join is reported once, the moment its second block appears.

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ai/pipestream/unlimitedocr/v1/unlimited_ocr.pb.h"

namespace uocr::output {

class ContinuationTracker {
 public:
  using Link = ::ai::pipestream::unlimitedocr::v1::ContinuationLink;
  using Block = ::ai::pipestream::unlimitedocr::v1::Block;

  // The links `block` completes, in order.
  std::vector<Link> add(const Block& block);

 private:
  struct Ref {
    std::uint32_t page_no = 0;
    std::uint32_t ordinal = 0;
  };

  bool table_chain_reaches(std::uint32_t tail_page, std::uint32_t page) const;

  std::map<std::string, Ref> open_;  // chain kind ("text", "table") -> its tail
  std::set<std::uint32_t> table_pages_;
  struct Previous {
    Ref ref;
    bool mergeable = false;
  };
  std::optional<Previous> previous_;
};

}  // namespace uocr::output
