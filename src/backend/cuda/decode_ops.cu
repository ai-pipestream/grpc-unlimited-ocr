#include "backend/cuda/decode_ops.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>

#include "backend/cuda/context_impl.h"

namespace uocr::cuda {

// Defined in ops.cu: the routing kernel, which takes an optional StepState.
void route_blocks_state(const Context& ctx, const float* z, std::int64_t max_blocks, const std::int64_t* block_min,
                        const std::int64_t* block_max, std::int64_t prompt_blocks, const StepState* state,
                        const RouteParams& params, float* mask);

namespace {

constexpr int kThreads = 256;
constexpr float kActClamp = 1.0e4f;

unsigned blocks_for(std::int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }
cudaStream_t stream_of(const Context& ctx) { return ctx.impl().stream; }

__global__ void embed_kernel(const __nv_bfloat16* table, const StepState* s, std::int64_t cols, float* dst) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < cols) dst[i] = __bfloat162float(table[s->token * cols + i]);
}

__global__ void rope_step_kernel(float* x, std::int64_t heads, std::int64_t dim, const StepState* s,
                                 const float* inv_tok, std::int64_t n_tok, const float* inv_page,
                                 std::int64_t page_mod) {
  const std::int64_t half = dim / 2;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= heads * half) return;
  const std::int64_t h = i / half, p = i % half;
  std::int64_t pg = s->page;
  if (page_mod > 0) pg = ((pg % page_mod) + page_mod) % page_mod;
  const float angle = p < n_tok ? static_cast<float>(s->tok) * inv_tok[p] : static_cast<float>(pg) * inv_page[p - n_tok];
  float sn, c;
  sincosf(angle, &sn, &c);
  float* v = x + h * dim;
  const float a = v[p], b = v[p + half];
  v[p] = a * c - b * sn;
  v[p + half] = b * c + a * sn;
}

__global__ void append_kernel(float* cache, const float* row, std::int64_t width, const StepState* s) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < width) cache[s->pos * width + i] = row[i];
}

// The router summary of the block holding pos: one warp per key for the
// head_cls logits, one thread for the (at most 32-wide) softmax, then one
// thread per feature for the weighted average.
__global__ void summary_step_kernel(const float* keys, std::int64_t d, const float* head_cls, std::int64_t block_size,
                                    const StepState* s, float* out) {
  const std::int64_t blk = s->pos / block_size;
  const std::int64_t first = blk * block_size, last = s->pos + 1;
  __shared__ float weight[64];
  __shared__ float total;
  const float inv_sqrt_d = rsqrtf(static_cast<float>(d));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  for (std::int64_t t = first + warp; t < last; t += warps) {
    float a = 0.0f;
    for (std::int64_t e = lane; e < d; e += 32) a += keys[t * d + e] * head_cls[e];
    for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffU, a, off);
    if (lane == 0) weight[t - first] = fminf(fmaxf(a * inv_sqrt_d, -kActClamp), kActClamp);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float peak = -FLT_MAX;
    for (std::int64_t t = first; t < last; ++t) peak = fmaxf(peak, weight[t - first]);
    float sum = 0.0f;
    for (std::int64_t t = first; t < last; ++t) {
      weight[t - first] = expf(weight[t - first] - peak);
      sum += weight[t - first];
    }
    total = fmaxf(sum, 1e-6f);
  }
  __syncthreads();
  for (std::int64_t e = threadIdx.x; e < d; e += blockDim.x) {
    float acc = 0.0f;
    for (std::int64_t t = first; t < last; ++t) acc += weight[t - first] / total * keys[t * d + e];
    out[blk * d + e] = acc;
  }
}

// One warp per block score.
__global__ void scores_step_kernel(const float* rq, const float* summaries, std::int64_t d, std::int64_t block_size,
                                   const StepState* s, float* z) {
  const std::int64_t nblk = s->pos / block_size + 1;
  const std::int64_t b = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (b >= nblk) return;
  float acc = 0.0f;
  for (std::int64_t e = lane; e < d; e += 32) acc += rq[e] * summaries[b * d + e];
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffU, acc, off);
  if (lane == 0) z[b] = acc * rsqrtf(static_cast<float>(d));
}

// Split-key decode attention. Grid (heads, kSplits): each block takes a
// contiguous slice of the keys, computes its masked scores (a warp per key),
// its local max, exp sum and value sum, and leaves them in scratch; a
// second kernel merges the slices per head. The slice length follows the
// key count read from the step state, so the launch shape never changes.
constexpr int kAttendThreads = 256;
constexpr int kSplits = 16;
constexpr int kMaxHeadDim = 128;

__global__ void attend_partial_kernel(const float* q, const float* k_cache, const float* v_cache, std::int64_t heads,
                                      std::int64_t hd, const float* mask, float scale, const StepState* s,
                                      float* scratch, std::int64_t capacity) {
  const std::int64_t h = blockIdx.x, split = blockIdx.y, keys = s->pos + 1, width = heads * hd;
  const std::int64_t chunk = (keys + kSplits - 1) / kSplits;
  const std::int64_t first = split * chunk, last = min(keys, first + chunk);
  float* sc = scratch + h * capacity;  // per-key scores of this head
  float* part = scratch + heads * capacity + (h * kSplits + split) * (hd + 2);  // [max, sum, acc[hd]]
  const float* qh = q + h * hd;
  __shared__ float red[kAttendThreads];
  __shared__ float partial[kAttendThreads];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  float peak = -INFINITY;
  for (std::int64_t j = first + warp; j < last; j += warps) {
    float v = -INFINITY;
    if (mask[j] != -INFINITY) {
      const float* kj = k_cache + j * width + h * hd;
      float dot = 0.0f;
      for (std::int64_t e = lane; e < hd; e += 32) dot += qh[e] * kj[e];
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffU, dot, off);  // every lane gets the sum
      v = dot * scale + mask[j];
    }
    if (lane == 0) sc[j] = v;
    peak = fmaxf(peak, v);
  }
  red[threadIdx.x] = peak;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
    __syncthreads();
  }
  peak = red[0];
  __syncthreads();
  float total = 0.0f;
  for (std::int64_t j = first + threadIdx.x; j < last; j += blockDim.x) {
    const float e = (peak == -INFINITY || sc[j] == -INFINITY) ? 0.0f : expf(sc[j] - peak);
    sc[j] = e;
    total += e;
  }
  red[threadIdx.x] = total;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
    __syncthreads();
  }
  const std::int64_t groups = blockDim.x / hd;
  const std::int64_t g = threadIdx.x / hd, e = threadIdx.x % hd;
  float acc = 0.0f;
  if (g < groups) {
    for (std::int64_t j = first + g; j < last; j += groups) {
      const float p = sc[j];
      if (p != 0.0f) acc += p * v_cache[j * width + h * hd + e];
    }
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  if (threadIdx.x < hd) {
    float sum = 0.0f;
    for (std::int64_t gi = 0; gi < groups; ++gi) sum += partial[gi * hd + threadIdx.x];
    part[2 + threadIdx.x] = sum;
  }
  if (threadIdx.x == 0) {
    part[0] = peak;
    part[1] = red[0];
  }
}

__global__ void attend_merge_kernel(std::int64_t heads, std::int64_t hd, const float* scratch, std::int64_t capacity,
                                    float* out) {
  const std::int64_t h = blockIdx.x, e = threadIdx.x;
  if (e >= hd) return;
  const float* parts = scratch + heads * capacity + h * kSplits * (hd + 2);
  float peak = -INFINITY;
  for (int i = 0; i < kSplits; ++i) peak = fmaxf(peak, parts[i * (hd + 2)]);
  float total = 0.0f, acc = 0.0f;
  for (int i = 0; i < kSplits; ++i) {
    const float* p = parts + i * (hd + 2);
    if (p[0] == -INFINITY) continue;
    const float w = expf(p[0] - peak);
    total += w * p[1];
    acc += w * p[2 + e];
  }
  out[h * hd + e] = acc / total;
}

}  // namespace

std::int64_t attend_scratch_floats(std::int64_t heads, std::int64_t hd, std::int64_t capacity) {
  return heads * capacity + heads * kSplits * (hd + 2);
}

void embed_step(const Context& ctx, const std::uint16_t* table, const StepState* state, std::int64_t cols, float* dst) {
  embed_kernel<<<blocks_for(cols), kThreads, 0, stream_of(ctx)>>>(reinterpret_cast<const __nv_bfloat16*>(table), state,
                                                                  cols, dst);
}

void rope_step(const Context& ctx, float* x, std::int64_t heads, std::int64_t dim, const StepState* state,
               const float* inv_tok, std::int64_t n_tok, const float* inv_page, std::int64_t page_mod) {
  rope_step_kernel<<<blocks_for(heads * dim / 2), kThreads, 0, stream_of(ctx)>>>(x, heads, dim, state, inv_tok, n_tok,
                                                                                 inv_page, page_mod);
}

void append_step(const Context& ctx, float* cache, const float* row, std::int64_t width, const StepState* state) {
  append_kernel<<<blocks_for(width), kThreads, 0, stream_of(ctx)>>>(cache, row, width, state);
}

void summary_step(const Context& ctx, const float* router_keys, std::int64_t d, const float* head_cls,
                  std::int64_t block_size, const StepState* state, float* summaries) {
  summary_step_kernel<<<1, 256, 0, stream_of(ctx)>>>(router_keys, d, head_cls, block_size, state, summaries);
}

void scores_step(const Context& ctx, const float* rq, const float* summaries, std::int64_t d, std::int64_t block_size,
                 const StepState* state, std::int64_t max_blocks, float* z) {
  // Launched for the capacity; warps past the current block count return.
  scores_step_kernel<<<blocks_for(max_blocks * 32), kThreads, 0, stream_of(ctx)>>>(rq, summaries, d, block_size, state,
                                                                                   z);
}

void route_step(const Context& ctx, const float* z, std::int64_t max_blocks, const std::int64_t* block_min,
                const std::int64_t* block_max, std::int64_t prompt_blocks, const StepState* state,
                const RouteParams& params, float* mask) {
  route_blocks_state(ctx, z, max_blocks, block_min, block_max, prompt_blocks, state, params, mask);
}

void attend_step(const Context& ctx, const float* q, const float* k_cache, const float* v_cache, std::int64_t heads,
                 std::int64_t hd, const float* mask, float scale, const StepState* state, float* scratch,
                 std::int64_t capacity, float* out) {
  if (hd > kMaxHeadDim) throw std::invalid_argument("attend_step: head dim above 128");
  attend_partial_kernel<<<dim3(static_cast<unsigned>(heads), kSplits), kAttendThreads, 0, stream_of(ctx)>>>(
      q, k_cache, v_cache, heads, hd, mask, scale, state, scratch, capacity);
  attend_merge_kernel<<<static_cast<unsigned>(heads), kMaxHeadDim, 0, stream_of(ctx)>>>(heads, hd, scratch, capacity,
                                                                                         out);
}

}  // namespace uocr::cuda
