#include "model/page_layout.h"

#include <algorithm>
#include <stdexcept>

namespace uocr {

PromptLayout layout_prompt(const std::vector<std::int64_t>& ids, const PageTokens& tokens) {
  const auto n = static_cast<std::int64_t>(ids.size());
  std::int64_t last_image = -1;
  for (std::int64_t i = 0; i < n; ++i) {
    if (ids[static_cast<std::size_t>(i)] == tokens.image) last_image = i;
  }
  std::vector<std::int64_t> cum(static_cast<std::size_t>(n));
  std::int64_t markers = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    markers += ids[static_cast<std::size_t>(i)] == tokens.page ? 1 : 0;
    cum[static_cast<std::size_t>(i)] = markers;
  }
  if (last_image < 0 || cum[static_cast<std::size_t>(last_image)] == 0) {
    throw std::invalid_argument("prompt needs image tokens with a <PAGE> marker before them");
  }
  PromptLayout out;
  out.prompt_pages = cum[static_cast<std::size_t>(last_image)];

  // Rope buckets: prefill positions take the marker count (1..N), later
  // positions count - N; 0 is the reserved "no page" bucket.
  std::vector<std::int64_t> bucket(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t c = cum[static_cast<std::size_t>(i)];
    bucket[static_cast<std::size_t>(i)] = std::max<std::int64_t>(i <= last_image ? c : c - out.prompt_pages, 0);
  }
  out.rope_pages = *std::ranges::max_element(bucket) + 1;
  std::vector<std::int64_t> running(static_cast<std::size_t>(out.rope_pages), 0);
  out.prefill_count.assign(static_cast<std::size_t>(out.rope_pages), 0);
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t b = bucket[static_cast<std::size_t>(i)];
    out.coords.push_back({running[static_cast<std::size_t>(b)]++, b});
    if (i <= last_image) out.prefill_count[static_cast<std::size_t>(b)]++;
  }

  // Page isolation ids (page-scoped prefill): image tokens and prefill
  // queries take marker count - 1, the instruction after the last image is
  // a global query (-1).
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t raw = cum[static_cast<std::size_t>(i)] - 1;
    const bool image = ids[static_cast<std::size_t>(i)] == tokens.image;
    out.image_page.push_back(image ? raw : -1);
    const std::int64_t q = i <= last_image ? raw : raw - out.prompt_pages;
    out.query_page.push_back(q < 0 ? -1 : q);
  }
  return out;
}

BlockPages block_image_pages(const PromptLayout& layout, std::int64_t block_size) {
  const auto n = static_cast<std::int64_t>(layout.image_page.size());
  const std::int64_t blocks = (n + block_size - 1) / block_size;
  BlockPages out{std::vector<std::int64_t>(static_cast<std::size_t>(blocks), -1),
                 std::vector<std::int64_t>(static_cast<std::size_t>(blocks), -1)};
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t page = layout.image_page[static_cast<std::size_t>(i)];
    if (page < 0) continue;
    auto& lo = out.min_page[static_cast<std::size_t>(i / block_size)];
    auto& hi = out.max_page[static_cast<std::size_t>(i / block_size)];
    lo = lo < 0 ? page : std::min(lo, page);
    hi = std::max(hi, page);
  }
  return out;
}

DecodeState::DecodeState(const PromptLayout& layout, const std::vector<std::int64_t>& prompt_ids,
                         const PageTokens& tokens)
    : layout_(layout), tokens_(tokens), coord_(layout.coords.back()) {
  markers_ = std::ranges::count(prompt_ids, tokens.page);
}

RopeCoord DecodeState::advance(std::int64_t token) {
  if (token == tokens_.page) {
    ++markers_;
    coord_.page = std::min(coord_.page + 1, layout_.rope_pages - 1);
    coord_.tok = layout_.prefill_count[static_cast<std::size_t>(std::max<std::int64_t>(coord_.page, 0))];
  } else {
    ++coord_.tok;
  }
  return coord_;
}

std::int64_t DecodeState::query_page() const {
  const std::int64_t page = markers_ - 1 - layout_.prompt_pages;
  return page >= 0 ? page : -1;
}

}  // namespace uocr
