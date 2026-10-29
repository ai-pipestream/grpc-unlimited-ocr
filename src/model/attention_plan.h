#pragma once

// The host-side plan of one sparse-attention call, shared by every
// backend so they attend the same keys: which blocks each query may route
// to, how raw block scores are masked, and the additive key mask that the
// selection becomes (0 in the local window, the block's bias on a selected
// block, -inf elsewhere).

#include <cstdint>
#include <vector>

#include "model/block_router.h"
#include "tensor/tensor.h"

namespace uocr {

// Per-query candidate lists (empty = all) flattened to [L, nblk], or empty
// when no query has a restriction.
std::vector<std::uint8_t> flatten_candidates(const std::vector<std::vector<std::uint8_t>>& candidates,
                                             std::int64_t nblk);

// Raw scores z [L, nblk] (already scaled by 1/sqrt(d)) clamped, with the
// blocks a query may not route to (its local window and after) masked.
void mask_block_scores(Tensor& z, std::int64_t first_pos, const RouterParams& params);

// The additive mask [L, keys] for queries at first_pos.. over keys 0..keys-1.
std::vector<float> key_mask(const BlockSelection& selection, std::int64_t rows, std::int64_t first_pos,
                            std::int64_t keys, const RouterParams& params);

}  // namespace uocr
