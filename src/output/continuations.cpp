#include "output/continuations.h"

namespace uocr::output {
namespace {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

// The chain a block can take part in: "text", "table", or none.
std::string chain_kind(const v1::Block& block) {
  if (block.kind() == v1::BLOCK_KIND_TEXT) return "text";
  if (block.kind() == v1::BLOCK_KIND_TABLE) return "table";
  return {};
}

// Upstream joins merged columns only between plain titles and text.
bool mergeable(const v1::Block& block) {
  return block.kind() == v1::BLOCK_KIND_TEXT ||
         (block.kind() == v1::BLOCK_KIND_TITLE && block.continuation() == v1::CONTINUATION_UNSPECIFIED);
}

v1::ContinuationLink link(std::uint32_t from_page, std::uint32_t from_ordinal, const v1::Block& to,
                          v1::ContinuationLinkKind kind) {
  v1::ContinuationLink out;
  out.set_from_page_no(from_page);
  out.set_from_ordinal(from_ordinal);
  out.set_to_page_no(to.page_no());
  out.set_to_ordinal(to.ordinal());
  out.set_kind(to.kind());
  out.set_link_kind(kind);
  return out;
}

}  // namespace

bool ContinuationTracker::table_chain_reaches(std::uint32_t tail_page, std::uint32_t page) const {
  if (page <= tail_page + 1) return true;
  const auto between = table_pages_.upper_bound(tail_page);
  return between == table_pages_.end() || *between >= page;
}

std::vector<ContinuationTracker::Link> ContinuationTracker::add(const Block& block) {
  std::vector<Link> links;
  const std::string chain = chain_kind(block);
  if (!chain.empty()) {
    if (block.continuation() == v1::CONTINUATION_CONTINUED_FROM_PREVIOUS) {
      const auto tail = open_.find(chain);
      if (tail != open_.end()) {
        if (chain == "table" && !table_chain_reaches(tail->second.page_no, block.page_no())) {
          open_.erase(tail);
        } else {
          links.push_back(
              link(tail->second.page_no, tail->second.ordinal, block, v1::CONTINUATION_LINK_KIND_CROSS_PAGE));
          tail->second = {block.page_no(), block.ordinal()};
        }
      }
    } else if (block.continuation() == v1::CONTINUATION_CONTINUES_ON_NEXT_PAGE || chain == "table") {
      open_[chain] = {block.page_no(), block.ordinal()};
    }
    if (chain == "table") table_pages_.insert(block.page_no());
  }
  if (block.merges_with_previous() && previous_.has_value() && previous_->ref.page_no == block.page_no() &&
      previous_->mergeable && mergeable(block)) {
    links.push_back(
        link(previous_->ref.page_no, previous_->ref.ordinal, block, v1::CONTINUATION_LINK_KIND_MERGE));
  }
  previous_ = Previous{{block.page_no(), block.ordinal()}, mergeable(block)};
  return links;
}

}  // namespace uocr::output
