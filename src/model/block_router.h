#pragma once

// Block selection for sparse attention. Keys are grouped into 32-token
// blocks; each block is summarised by a learned query (head_cls) over its
// router keys, every query scores the blocks that lie wholly before its
// local window, and a 1.5-entmax over the best 16 scores picks up to 12
// blocks with a log-prior bias. The local window (the last 128 positions)
// is always attended on top.

#include <cstdint>
#include <vector>

#include "tensor/tensor.h"

namespace uocr {

struct RouterParams {
  std::int64_t block_size = 32;
  std::int64_t local_blocks = 4;  // local window = 128 positions
  std::int64_t top_m = 16;        // entmax window
  std::int64_t top_k = 12;        // blocks read
  float idx_scale = 0.0f;         // set from scaling_factor / sqrt(index_dim)
  float floor = 1e-4f;
  float sigma = 1.0f;

  std::int64_t local_window() const { return local_blocks * block_size; }
};

// For each query, up to top_k blocks and the bias each adds to its keys'
// logits. `valid` is false for padding selections that are not attended.
struct BlockSelection {
  std::vector<std::int64_t> block;  // [queries, top_k]
  std::vector<float> bias;          // [queries, top_k]
  std::vector<std::uint8_t> valid;  // [queries, top_k]
  std::int64_t width = 0;           // top_k actually used (<= params.top_k)
};

// head_cls summaries of each block of router keys idx_k [S, d]: [nblk, d].
Tensor block_summaries(const Tensor& idx_k, const Tensor& head_cls, std::int64_t block_size);

// Block scores [Lq, nblk] for router queries idx_q [Lq, d] at absolute
// positions q_pos; blocks overlapping the query's local window, and blocks
// past it, score kScoreMasked.
Tensor block_scores(const Tensor& idx_q, const Tensor& summaries, const std::vector<std::int64_t>& q_pos,
                    std::int64_t keys, const RouterParams& params);

// Routes each row of `scores` within its candidate domain (`candidate`
// [Lq, nblk], empty = every block), falling back to every scoreable block
// when a row has no candidate.
BlockSelection select_blocks(const Tensor& scores, const std::vector<std::uint8_t>& candidate,
                             const RouterParams& params);

// 1.5-entmax of one row, in place (the sparse alternative to softmax).
void entmax15(std::vector<float>& x);

inline constexpr float kScoreMasked = -1.0e9f;

}  // namespace uocr
