#pragma once

// Ops for one decode step whose shapes do not depend on how many tokens
// came before: every length-dependent value (position, key count, block
// count, the fed token, its rope coordinate, its page) is read from a
// StepState on the device. A whole decode step built from these and
// fixed-shape ops can be captured once as a CUDA graph and replayed for
// every token, with only the StepState rewritten between replays.

#include <cstdint>

#include "backend/cuda/context.h"
#include "backend/cuda/ops.h"

namespace uocr::cuda {

struct StepState {
  std::int64_t token;       // the token being fed
  std::int64_t pos;         // its position; keys = pos + 1 after the append
  std::int64_t tok;         // rope token coordinate
  std::int64_t page;        // rope page coordinate
  std::int64_t query_page;  // response page for routing, -1 = unrestricted
};

void embed_step(const Context& ctx, const std::uint16_t* table, const StepState* state, std::int64_t cols, float* dst);
void rope_step(const Context& ctx, float* x, std::int64_t heads, std::int64_t dim, const StepState* state,
               const float* inv_tok, std::int64_t n_tok, const float* inv_page, std::int64_t page_mod);
// cache[pos] = row, rows `width` wide.
void append_step(const Context& ctx, float* cache, const float* row, std::int64_t width, const StepState* state);
// Refreshes the router summary of the block holding pos (the only one a
// new key changes).
void summary_step(const Context& ctx, const float* router_keys, std::int64_t d, const float* head_cls,
                  std::int64_t block_size, const StepState* state, float* summaries);
// z[b] = rq . summaries[b] / sqrt(d) for every block up to the one holding pos.
void scores_step(const Context& ctx, const float* rq, const float* summaries, std::int64_t d, std::int64_t block_size,
                 const StepState* state, std::int64_t max_blocks, float* z);
// route_blocks for the one decode row; `max_blocks` sizes the kernel's
// scratch and must cover the cache capacity.
void route_step(const Context& ctx, const float* z, std::int64_t max_blocks, const std::int64_t* block_min,
                const std::int64_t* block_max, std::int64_t prompt_blocks, const StepState* state,
                const RouteParams& params, float* mask);
// Masked attention of one query over keys 0..pos: out [heads * hd].
// `scratch` holds attend_scratch_floats(heads, hd, capacity) floats.
std::int64_t attend_scratch_floats(std::int64_t heads, std::int64_t hd, std::int64_t capacity);
void attend_step(const Context& ctx, const float* q, const float* k_cache, const float* v_cache, std::int64_t heads,
                 std::int64_t hd, const float* mask, float scale, const StepState* state, float* scratch,
                 std::int64_t capacity, float* out);

}  // namespace uocr::cuda
