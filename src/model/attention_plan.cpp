#include "model/attention_plan.h"

#include <algorithm>
#include <limits>

namespace uocr {

std::vector<std::uint8_t> flatten_candidates(const std::vector<std::vector<std::uint8_t>>& candidates,
                                             std::int64_t nblk) {
  const bool any = std::ranges::any_of(candidates, [](const auto& c) { return !c.empty(); });
  if (!any) return {};
  const auto l = static_cast<std::int64_t>(candidates.size());
  std::vector<std::uint8_t> out(static_cast<std::size_t>(l * nblk), 1);
  for (std::int64_t i = 0; i < l; ++i) {
    const auto& c = candidates[static_cast<std::size_t>(i)];
    if (c.empty()) continue;
    for (std::int64_t b = 0; b < nblk; ++b) {
      out[static_cast<std::size_t>(i * nblk + b)] = b < static_cast<std::int64_t>(c.size()) ? c[static_cast<std::size_t>(b)] : 0;
    }
  }
  return out;
}

void mask_block_scores(Tensor& z, std::int64_t first_pos, const RouterParams& params) {
  constexpr float kActClamp = 1.0e4f;
  const std::int64_t nblk = z.dim(1);
  for (std::int64_t i = 0; i < z.dim(0); ++i) {
    const std::int64_t last_remote = (first_pos + i) / params.block_size - params.local_blocks;
    float* row = z.row(i);
    for (std::int64_t b = 0; b < nblk; ++b) {
      row[b] = b <= last_remote ? std::clamp(row[b], -kActClamp, kActClamp) : kScoreMasked;
    }
  }
}

std::vector<float> key_mask(const BlockSelection& sel, std::int64_t rows, std::int64_t first_pos, std::int64_t keys,
                            const RouterParams& params) {
  constexpr float kOff = -std::numeric_limits<float>::infinity();
  const std::int64_t nblk = (keys + params.block_size - 1) / params.block_size;
  const std::int64_t window = params.local_window();
  std::vector<float> mask(static_cast<std::size_t>(rows * keys), kOff);
  std::vector<float> block_bias(static_cast<std::size_t>(nblk));
  for (std::int64_t i = 0; i < rows; ++i) {
    std::ranges::fill(block_bias, kOff);
    for (std::int64_t j = 0; j < sel.width; ++j) {
      const std::size_t slot = static_cast<std::size_t>(i * sel.width + j);
      if (sel.valid[slot] != 0) block_bias[static_cast<std::size_t>(sel.block[slot])] = sel.bias[slot];
    }
    const std::int64_t pos = first_pos + i;
    float* row = mask.data() + i * keys;
    for (std::int64_t j = 0; j <= pos && j < keys; ++j) {
      row[j] = j > pos - window ? 0.0f : block_bias[static_cast<std::size_t>(j / params.block_size)];
    }
  }
  return mask;
}

}  // namespace uocr
