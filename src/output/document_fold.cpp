#include "output/document_fold.h"

#include <map>
#include <numeric>
#include <utility>

namespace uocr::output {
namespace {

// Union-find over block indices; the smaller index stays the root, so a
// group's root is its first block in reading order.
class Groups {
 public:
  explicit Groups(std::size_t n) : parent_(n) { std::iota(parent_.begin(), parent_.end(), 0); }

  std::size_t find(std::size_t x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }

  void join(std::size_t a, std::size_t b) {
    const std::size_t ra = find(a);
    const std::size_t rb = find(b);
    if (ra != rb) parent_[std::max(ra, rb)] = std::min(ra, rb);
  }

 private:
  std::vector<std::size_t> parent_;
};

}  // namespace

docv1::Document page_document(const DocumentInput& input, const std::vector<v1::Block>& blocks) {
  DocumentBuilder builder(input);
  for (const v1::Block& block : blocks) builder.add_group({&block});
  return builder.finish();
}

docv1::Document fold_document(const DocumentInput& input, const std::vector<v1::Block>& blocks,
                              const std::vector<v1::ContinuationLink>& links) {
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> index;
  for (std::size_t i = 0; i < blocks.size(); ++i) index[{blocks[i].page_no(), blocks[i].ordinal()}] = i;
  Groups groups(blocks.size());
  for (const v1::ContinuationLink& link : links) {
    const auto from = index.find({link.from_page_no(), link.from_ordinal()});
    const auto to = index.find({link.to_page_no(), link.to_ordinal()});
    if (from != index.end() && to != index.end()) groups.join(from->second, to->second);
  }
  std::map<std::size_t, std::vector<const v1::Block*>> members;
  for (std::size_t i = 0; i < blocks.size(); ++i) members[groups.find(i)].push_back(&blocks[i]);
  DocumentBuilder builder(input);
  for (const auto& [root, group] : members) builder.add_group(group);
  return builder.finish();
}

}  // namespace uocr::output
