#include "backend/cuda/gemv_ops.h"

#include <cuda_bf16.h>

#include <stdexcept>

#include "backend/cuda/context_impl.h"
#include "backend/cuda/ops.h"

namespace uocr::cuda {
namespace {

constexpr int kWarp = 32;
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / kWarp;

cudaStream_t stream_of(const Context& ctx) { return ctx.impl().stream; }

__device__ float warp_sum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffU, v, off);
  return v;
}

__device__ void stage_bf16(const float* x, std::int64_t k, float* xs) {
  for (std::int64_t i = threadIdx.x; i < k; i += blockDim.x) xs[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
  __syncthreads();
}

// Row `row` of a weight matrix dotted with the staged input, by one warp.
// Eight bf16 weights against eight staged inputs. The inputs are read as two
// float4s: per-float reads at an eight-float lane stride would hit the same
// shared-memory banks.
__device__ float dot8(const uint4& packed, const float* xv) {
  const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&packed);
  const float4 x0 = reinterpret_cast<const float4*>(xv)[0];
  const float4 x1 = reinterpret_cast<const float4*>(xv)[1];
  const float2 f0 = __bfloat1622float2(pair[0]), f1 = __bfloat1622float2(pair[1]);
  const float2 f2 = __bfloat1622float2(pair[2]), f3 = __bfloat1622float2(pair[3]);
  return f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
}

struct Bf16Rows {
  const __nv_bfloat16* w;
  // Two rows against the same input in one pass: both weight streams are in
  // flight at once.
  __device__ void dot2(std::int64_t ra, std::int64_t rb, std::int64_t k, const float* xs, int lane, float& a,
                       float& b) const {
    if (k % 8 != 0) {
      a = dot(ra, k, xs, lane);
      b = dot(rb, k, xs, lane);
      return;
    }
    const uint4* wa = reinterpret_cast<const uint4*>(w + ra * k);
    const uint4* wb = reinterpret_cast<const uint4*>(w + rb * k);
    float sa = 0.0f, sb = 0.0f;
    for (std::int64_t c = lane; c < k / 8; c += kWarp) {
      const uint4 pa = wa[c], pb = wb[c];
      sa += dot8(pa, xs + c * 8);
      sb += dot8(pb, xs + c * 8);
    }
    a = warp_sum(sa);
    b = warp_sum(sb);
  }
  __device__ float dot(std::int64_t row, std::int64_t k, const float* xs, int lane) const {
    const __nv_bfloat16* wr = w + row * k;
    float acc = 0.0f;
    if (k % 8 == 0) {
      const uint4* w8 = reinterpret_cast<const uint4*>(wr);
      for (std::int64_t c = lane; c < k / 8; c += kWarp) acc += dot8(w8[c], xs + c * 8);
    } else {
      for (std::int64_t i = lane; i < k; i += kWarp) acc += __bfloat162float(wr[i]) * xs[i];
    }
    return warp_sum(acc);
  }
};

struct Q8Rows {
  const std::int8_t* q;
  const float* s;
  __device__ void dot2(std::int64_t ra, std::int64_t rb, std::int64_t k, const float* xs, int lane, float& a,
                       float& b) const {
    a = dot(ra, k, xs, lane);
    b = dot(rb, k, xs, lane);
  }
  // Sixteen weights per load; a load never straddles a 32-wide scale group.
  __device__ float dot(std::int64_t row, std::int64_t k, const float* xs, int lane) const {
    const int4* q16 = reinterpret_cast<const int4*>(q + row * k);
    const float* sr = s + row * (k / kQ8Group);
    float acc = 0.0f;
    for (std::int64_t c = lane; c < k / 16; c += kWarp) {
      const int4 packed = q16[c];
      const std::int8_t* b = reinterpret_cast<const std::int8_t*>(&packed);
      const float4* xv = reinterpret_cast<const float4*>(xs + c * 16);
      float part = 0.0f;
      for (int v = 0; v < 4; ++v) {
        const float4 xq = xv[v];
        part += static_cast<float>(b[4 * v]) * xq.x + static_cast<float>(b[4 * v + 1]) * xq.y +
                static_cast<float>(b[4 * v + 2]) * xq.z + static_cast<float>(b[4 * v + 3]) * xq.w;
      }
      acc += part * sr[(c * 16) / kQ8Group];
    }
    return warp_sum(acc);
  }
};

// gate . x and up . x for the same row of two stacks, in one pass for bf16.
__device__ void dot_pair(const Bf16Rows& a, const Bf16Rows& b, std::int64_t r, std::int64_t k, const float* xs, int lane,
                         float& out_a, float& out_b) {
  if (k % 8 != 0) {
    out_a = a.dot(r, k, xs, lane);
    out_b = b.dot(r, k, xs, lane);
    return;
  }
  const uint4* wa = reinterpret_cast<const uint4*>(a.w + r * k);
  const uint4* wb = reinterpret_cast<const uint4*>(b.w + r * k);
  float sa = 0.0f, sb = 0.0f;
  for (std::int64_t c = lane; c < k / 8; c += kWarp) {
    const uint4 pa = wa[c], pb = wb[c];
    sa += dot8(pa, xs + c * 8);
    sb += dot8(pb, xs + c * 8);
  }
  out_a = warp_sum(sa);
  out_b = warp_sum(sb);
}

__device__ void dot_pair(const Q8Rows& a, const Q8Rows& b, std::int64_t r, std::int64_t k, const float* xs, int lane,
                         float& out_a, float& out_b) {
  out_a = a.dot(r, k, xs, lane);
  out_b = b.dot(r, k, xs, lane);
}

template <typename Rows>
__global__ void gemv_kernel(const float* x, std::int64_t k, Rows w, std::int64_t n, const float* bias, float* y) {
  extern __shared__ float xs[];
  stage_bf16(x, k, xs);
  const int lane = threadIdx.x % kWarp;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kWarps + threadIdx.x / kWarp;
  if (row >= n) return;
  const float acc = w.dot(row, k, xs, lane);
  if (lane == 0) y[row] = acc + (bias != nullptr ? bias[row] : 0.0f);
}

// Grid (ffn / warps, k slots): gate and up rows of one routed expert.
template <typename Rows>
__global__ void expert_gate_up_kernel(const float* x, std::int64_t hidden, std::int64_t ffn, Rows gate, Rows up,
                                      const std::int64_t* ids, float* act) {
  extern __shared__ float xs[];
  stage_bf16(x, hidden, xs);
  const int lane = threadIdx.x % kWarp;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kWarps + threadIdx.x / kWarp;
  const std::int64_t slot = blockIdx.y;
  if (row >= ffn) return;
  const std::int64_t r = ids[slot] * ffn + row;
  float g = 0.0f, u = 0.0f;
  dot_pair(gate, up, r, hidden, xs, lane, g, u);
  if (lane == 0) act[slot * ffn + row] = __bfloat162float(__float2bfloat16_rn(g / (1.0f + expf(-g)) * u));
}

template <typename Rows>
__global__ void expert_down_kernel(const float* act, std::int64_t hidden, std::int64_t ffn, Rows down,
                                   const std::int64_t* ids, const float* weights, std::int64_t k, float* out) {
  const float* acts = act;  // small and shared by every block: served from cache
  const int lane = threadIdx.x % kWarp;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * kWarps + threadIdx.x / kWarp;
  if (row >= hidden) return;
  float total = 0.0f;
  for (std::int64_t j = 0; j < k; ++j) total += weights[j] * down.dot(ids[j] * hidden + row, ffn, acts + j * ffn, lane);
  if (lane == 0) out[row] = total;
}

template <typename Rows>
void expert_ffn(const Context& ctx, const float* x, std::int64_t hidden, std::int64_t ffn, Rows gate, Rows up,
                Rows down, const std::int64_t* ids, const float* weights, std::int64_t k, float* out) {
  auto stream = stream_of(ctx);
  float* act = nullptr;
  check(cudaMallocAsync(&act, static_cast<std::size_t>(k * ffn) * 4, stream), "expert scratch");
  const dim3 grid(static_cast<unsigned>((ffn + kWarps - 1) / kWarps), static_cast<unsigned>(k));
  expert_gate_up_kernel<<<grid, kBlock, static_cast<std::size_t>(hidden) * 4, stream>>>(x, hidden, ffn, gate, up, ids,
                                                                                       act);
  expert_down_kernel<<<static_cast<unsigned>((hidden + kWarps - 1) / kWarps), kBlock, 0, stream>>>(act, hidden, ffn, down,
                                                                                                ids, weights, k, out);
  check(cudaFreeAsync(act, stream), "expert scratch free");
}

__global__ void dequant_kernel(const std::int8_t* q, const float* s, std::int64_t rows, std::int64_t k,
                               __nv_bfloat16* out) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * k) return;
  const std::int64_t r = i / k, c = i % k;
  out[i] = __float2bfloat16_rn(static_cast<float>(q[i]) * s[r * (k / kQ8Group) + c / kQ8Group]);
}

void require_q8_shape(std::int64_t k) {
  if (k % kQ8Group != 0) throw std::invalid_argument("int8 weights need an input width that is a multiple of 32");
}

}  // namespace

void gemv_bf16(const Context& ctx, const float* x, std::int64_t k, const std::uint16_t* w, std::int64_t n,
               const float* bias, float* y) {
  gemv_kernel<<<static_cast<unsigned>((n + kWarps - 1) / kWarps), kBlock, static_cast<std::size_t>(k) * 4,
                stream_of(ctx)>>>(x, k, Bf16Rows{reinterpret_cast<const __nv_bfloat16*>(w)}, n, bias, y);
}

void linear_q8(const Context& ctx, const float* x, std::int64_t m, std::int64_t k, Q8Weights w, std::int64_t n,
               const float* bias, float* y) {
  require_q8_shape(k);
  auto stream = stream_of(ctx);
  if (m == 1 && k <= kGemvMaxK) {
    gemv_kernel<<<static_cast<unsigned>((n + kWarps - 1) / kWarps), kBlock, static_cast<std::size_t>(k) * 4, stream>>>(
        x, k, Q8Rows{w.q, w.scales}, n, bias, y);
    return;
  }
  __nv_bfloat16* wb = nullptr;
  check(cudaMallocAsync(&wb, static_cast<std::size_t>(n * k) * 2, stream), "dequant scratch");
  dequant_kernel<<<static_cast<unsigned>((n * k + 255) / 256), 256, 0, stream>>>(w.q, w.scales, n, k, wb);
  linear(ctx, x, m, k, reinterpret_cast<const std::uint16_t*>(wb), n, bias, y);
  check(cudaFreeAsync(wb, stream), "dequant scratch free");
}

void expert_ffn_one(const Context& ctx, const float* x, std::int64_t hidden, std::int64_t ffn,
                    const std::uint16_t* gate, const std::uint16_t* up, const std::uint16_t* down,
                    const std::int64_t* ids, const float* weights, std::int64_t k, float* out) {
  auto rows = [](const std::uint16_t* p) { return Bf16Rows{reinterpret_cast<const __nv_bfloat16*>(p)}; };
  expert_ffn(ctx, x, hidden, ffn, rows(gate), rows(up), rows(down), ids, weights, k, out);
}

void expert_ffn_one_q8(const Context& ctx, const float* x, std::int64_t hidden, std::int64_t ffn, Q8Weights gate,
                       Q8Weights up, Q8Weights down, const std::int64_t* ids, const float* weights, std::int64_t k,
                       float* out) {
  require_q8_shape(hidden);
  require_q8_shape(ffn);
  expert_ffn(ctx, x, hidden, ffn, Q8Rows{gate.q, gate.scales}, Q8Rows{up.q, up.scales}, Q8Rows{down.q, down.scales},
             ids, weights, k, out);
}

}  // namespace uocr::cuda
