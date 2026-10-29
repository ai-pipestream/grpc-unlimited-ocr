#include "backend/cuda/ops.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <climits>
#include <stdexcept>
#include <cmath>

#include "backend/cuda/context_impl.h"
#include "backend/cuda/decode_ops.h"

namespace uocr::cuda {
namespace {

constexpr int kThreads = 256;
constexpr float kActClamp = 1.0e4f;

unsigned blocks_for(std::int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }
cudaStream_t stream_of(const Context& ctx) { return ctx.impl().stream; }

__global__ void to_bf16_kernel(const float* x, __nv_bfloat16* y, std::int64_t n) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

// y[r] = w[r] . bf16(x) for one input row: a warp per output row, eight
// bf16 weights per load, x staged once per block in shared memory.
__global__ void gemv_bf16_kernel(const float* x, std::int64_t k, const __nv_bfloat16* w, std::int64_t n,
                                 const float* bias, float* y) {
  extern __shared__ float xs[];
  for (std::int64_t i = threadIdx.x; i < k; i += blockDim.x) xs[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
  __syncthreads();
  const int lane = threadIdx.x % 32, warps = blockDim.x / 32;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + threadIdx.x / 32;
  if (row >= n) return;
  const __nv_bfloat16* wr = w + row * k;
  float acc = 0.0f;
  if (k % 8 == 0) {
    const uint4* w8 = reinterpret_cast<const uint4*>(wr);
    for (std::int64_t c = lane; c < k / 8; c += 32) {
      const uint4 packed = w8[c];
      const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&packed);
      const float* xv = xs + c * 8;
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(pair[q]);
        acc += f.x * xv[2 * q] + f.y * xv[2 * q + 1];
      }
    }
  } else {
    for (std::int64_t i = lane; i < k; i += 32) acc += __bfloat162float(wr[i]) * xs[i];
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffU, acc, off);
  if (lane == 0) y[row] = acc + (bias != nullptr ? bias[row] : 0.0f);
}

__global__ void add_bias_kernel(float* y, const float* bias, std::int64_t rows, std::int64_t cols) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < rows * cols) y[i] += bias[i % cols];
}

__global__ void add_kernel(float* y, const float* x, std::int64_t n) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) y[i] += x[i];
}

// One block per row; the row is reduced in fp32 with a tree, then scaled.
template <int kMode>  // 0 layer norm, 1 rms norm, 2 router norm
__global__ void norm_kernel(float* x, const float* w, const float* b, std::int64_t cols, float eps) {
  __shared__ float s1[kThreads];
  __shared__ float s2[kThreads];
  float* row = x + blockIdx.x * cols;
  float a = 0.0f, q = 0.0f;
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = row[i];
    if (kMode == 2 && !isfinite(v)) {
      v = 0.0f;
      row[i] = 0.0f;
    }
    a += v;
    q += v * v;
  }
  s1[threadIdx.x] = a;
  s2[threadIdx.x] = q;
  __syncthreads();
  for (int step = blockDim.x / 2; step > 0; step >>= 1) {
    if (threadIdx.x < step) {
      s1[threadIdx.x] += s1[threadIdx.x + step];
      s2[threadIdx.x] += s2[threadIdx.x + step];
    }
    __syncthreads();
  }
  const float n = static_cast<float>(cols);
  if (kMode == 0) {
    const float mean = s1[0] / n;
    const float var = fmaxf(s2[0] / n - mean * mean, 0.0f);
    const float inv = rsqrtf(var + eps);
    for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) row[i] = (row[i] - mean) * inv * w[i] + b[i];
  } else {
    const float inv = rsqrtf(s2[0] / n + eps);
    for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
      if (kMode == 1) {
        row[i] = w[i] * (row[i] * inv);
      } else {
        const float v = fminf(fmaxf(row[i] * inv, -kActClamp), kActClamp);
        row[i] = fminf(fmaxf(v * (1.0f + w[i]), -kActClamp), kActClamp);
      }
    }
  }
}

template <int kMode>  // 0 exact gelu, 1 quick gelu
__global__ void act_kernel(float* x, std::int64_t n) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  x[i] = kMode == 0 ? 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)) : v / (1.0f + expf(-1.702f * v));
}

__global__ void silu_mul_kernel(float* g, const float* u, std::int64_t n) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

__global__ void softmax_kernel(float* x, std::int64_t cols) {
  __shared__ float red[kThreads];
  float* row = x + blockIdx.x * cols;
  float peak = -INFINITY;
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) peak = fmaxf(peak, row[i]);
  red[threadIdx.x] = peak;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  peak = red[0];
  __syncthreads();
  if (peak == -INFINITY) {
    for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) row[i] = 0.0f;
    return;
  }
  float total = 0.0f;
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = expf(row[i] - peak);
    row[i] = e;
    total += e;
  }
  red[threadIdx.x] = total;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = 1.0f / red[0];
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) row[i] *= inv;
}

__global__ void scale_mask_kernel(float* x, float scale, const float* mask, std::int64_t per, std::int64_t total) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < total) x[i] = x[i] * scale + (mask != nullptr ? mask[i % per] : 0.0f);
}

__global__ void rope_kernel(float* x, std::int64_t rows, std::int64_t heads, std::int64_t dim, const std::int64_t* tok,
                            const std::int64_t* page, const float* inv_tok, std::int64_t n_tok, const float* inv_page,
                            std::int64_t page_mod, std::int64_t one_tok, std::int64_t one_page) {
  const std::int64_t half = dim / 2;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * heads * half) return;
  const std::int64_t r = i / (heads * half), h = (i / half) % heads, p = i % half;
  std::int64_t pg = page != nullptr ? page[r] : one_page;
  const std::int64_t tk = tok != nullptr ? tok[r] : one_tok;
  if (page_mod > 0) pg = ((pg % page_mod) + page_mod) % page_mod;
  const float angle = p < n_tok ? static_cast<float>(tk) * inv_tok[p] : static_cast<float>(pg) * inv_page[p - n_tok];
  float s, c;
  sincosf(angle, &s, &c);
  float* v = x + r * heads * dim + h * dim;
  const float a = v[p], b = v[p + half];
  v[p] = a * c - b * s;
  v[p + half] = b * c + a * s;
}

__global__ void im2col_kernel(const float* x, std::int64_t c, std::int64_t h, std::int64_t w, std::int64_t kh,
                              std::int64_t kw, int stride, int pad, std::int64_t oh, std::int64_t ow, float* cols) {
  const std::int64_t patch = c * kh * kw;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= oh * ow * patch) return;
  const std::int64_t pos = i / patch, e = i % patch;
  const std::int64_t oy = pos / ow, ox = pos % ow;
  const std::int64_t ci = e / (kh * kw), ky = (e / kw) % kh, kx = e % kw;
  const std::int64_t iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
  cols[i] = (iy >= 0 && iy < h && ix >= 0 && ix < w) ? x[(ci * h + iy) * w + ix] : 0.0f;
}

__global__ void transpose_kernel(const float* x, std::int64_t a, std::int64_t b, float* y) {
  // x [a, b] -> y [b, a]
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < a * b) y[(i % b) * a + i / b] = x[i];
}

__global__ void gather_kernel(const float* src, const std::int64_t* index, std::int64_t rows, std::int64_t cols,
                              float* dst) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < rows * cols) dst[i] = src[index[i / cols] * cols + i % cols];
}

__global__ void gather_bf16_kernel(const __nv_bfloat16* table, const std::int64_t* index, std::int64_t rows,
                                   std::int64_t cols, float* dst) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < rows * cols) dst[i] = __bfloat162float(table[index[i / cols] * cols + i % cols]);
}

// One thread per column; rows are applied in order, so the sum order is
// fixed whatever the launch shape.
__global__ void scatter_add_kernel(float* dst, const std::int64_t* index, const float* weight, const float* src,
                                   std::int64_t rows, std::int64_t cols) {
  const std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (c >= cols) return;
  for (std::int64_t r = 0; r < rows; ++r) dst[index[r] * cols + c] += weight[r] * src[r * cols + c];
}

// Decomposed: first q . rh and q . rw per query (n * (h + w) dots), then
// the n * n bias as sums of the two.
__global__ void rel_dot_kernel(const float* q, std::int64_t ldq, std::int64_t h, std::int64_t w, std::int64_t d,
                               const float* rh, const float* rw, float* dots) {
  const std::int64_t n = h * w, span = h + w;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n * span) return;
  const std::int64_t t = i / span, j = i % span;
  const float* qt = q + t * ldq;
  const float* r = j < h ? rh + ((t / w) * h + j) * d : rw + ((t % w) * w + (j - h)) * d;
  float s = 0.0f;
  for (std::int64_t e = 0; e < d; ++e) s += qt[e] * r[e];
  dots[i] = s;
}

__global__ void rel_expand_kernel(const float* dots, std::int64_t h, std::int64_t w, float* bias) {
  const std::int64_t n = h * w, span = h + w;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n * n) return;
  const std::int64_t t = i / n, key = i % n;
  bias[i] = dots[t * span + key / w] + dots[t * span + h + key % w];
}

__global__ void block_summary_kernel(const float* keys, std::int64_t s, std::int64_t d, const float* head_cls,
                                     std::int64_t block_size, float* out) {
  // One block of threads per key block; thread e owns feature e.
  const std::int64_t blk = blockIdx.x;
  const std::int64_t first = blk * block_size;
  const std::int64_t last = min(first + block_size, s);
  __shared__ float weight[64];
  __shared__ float total;
  const float inv_sqrt_d = rsqrtf(static_cast<float>(d));
  if (threadIdx.x == 0) {
    float peak = -FLT_MAX;
    for (std::int64_t t = first; t < last; ++t) {
      float a = 0.0f;
      for (std::int64_t e = 0; e < d; ++e) a += keys[t * d + e] * head_cls[e];
      a = fminf(fmaxf(a * inv_sqrt_d, -kActClamp), kActClamp);
      weight[t - first] = a;
      peak = fmaxf(peak, a);
    }
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

__global__ void concat_kernel(const float* a, std::int64_t ca, const float* b, std::int64_t cb, std::int64_t rows,
                              float* out) {
  const std::int64_t w = ca + cb;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * w) return;
  const std::int64_t r = i / w, c = i % w;
  out[i] = c < ca ? a[r * ca + c] : b[r * cb + (c - ca)];
}

}  // namespace

void concat_cols(const Context& ctx, const float* a, std::int64_t ca, const float* b, std::int64_t cb, std::int64_t rows,
                 float* out) {
  concat_kernel<<<blocks_for(rows * (ca + cb)), kThreads, 0, stream_of(ctx)>>>(a, ca, b, cb, rows, out);
}

void copy(const Context& ctx, float* dst, const float* src, std::int64_t n) {
  check(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(n) * 4, cudaMemcpyDeviceToDevice, stream_of(ctx)), "copy");
}

void linear(const Context& ctx, const float* x, std::int64_t m, std::int64_t k, const std::uint16_t* w_bf16,
            std::int64_t n, const float* bias, float* y) {
  auto stream = stream_of(ctx);
  if (m == 1 && k <= 12288) {
    constexpr int kBlock = 256, kWarps = kBlock / 32;
    gemv_bf16_kernel<<<static_cast<unsigned>((n + kWarps - 1) / kWarps), kBlock, static_cast<std::size_t>(k) * 4,
                       stream>>>(x, k, reinterpret_cast<const __nv_bfloat16*>(w_bf16), n, bias, y);
    return;
  }
  __nv_bfloat16* xb = nullptr;
  check(cudaMallocAsync(&xb, static_cast<std::size_t>(m * k) * 2, stream), "linear scratch");
  to_bf16_kernel<<<blocks_for(m * k), kThreads, 0, stream>>>(x, xb, m * k);
  const float one = 1.0f, zero = 0.0f;
  // Row-major y = x w^T is column-major y^T = w x^T.
  check(cublasGemmEx(ctx.impl().blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                     static_cast<int>(k), &one, w_bf16, CUDA_R_16BF, static_cast<int>(k), xb, CUDA_R_16BF,
                     static_cast<int>(k), &zero, y, CUDA_R_32F, static_cast<int>(n), CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
  check(cudaFreeAsync(xb, stream), "linear scratch free");
  if (bias != nullptr) add_bias_kernel<<<blocks_for(m * n), kThreads, 0, stream>>>(y, bias, m, n);
}

void gemm_nt(const Context& ctx, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
             std::int64_t lda, std::int64_t stride_a, const float* b, std::int64_t ldb, std::int64_t stride_b, float* c,
             std::int64_t ldc, std::int64_t stride_c, std::int64_t batch) {
  const float zero = 0.0f;
  check(cublasSgemmStridedBatched(ctx.impl().blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                  static_cast<int>(k), &alpha, b, static_cast<int>(ldb), stride_b, a,
                                  static_cast<int>(lda), stride_a, &zero, c, static_cast<int>(ldc), stride_c,
                                  static_cast<int>(batch)),
        "cublasSgemm nt");
}

void gemm_nn(const Context& ctx, std::int64_t m, std::int64_t n, std::int64_t k, const float* a, std::int64_t lda,
             std::int64_t stride_a, const float* b, std::int64_t ldb, std::int64_t stride_b, float* c, std::int64_t ldc,
             std::int64_t stride_c, std::int64_t batch) {
  const float one = 1.0f, zero = 0.0f;
  check(cublasSgemmStridedBatched(ctx.impl().blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                                  static_cast<int>(k), &one, b, static_cast<int>(ldb), stride_b, a,
                                  static_cast<int>(lda), stride_a, &zero, c, static_cast<int>(ldc), stride_c,
                                  static_cast<int>(batch)),
        "cublasSgemm nn");
}

void add_inplace(const Context& ctx, float* y, const float* x, std::int64_t n) {
  add_kernel<<<blocks_for(n), kThreads, 0, stream_of(ctx)>>>(y, x, n);
}

void add_rows(const Context& ctx, float* y, const float* row, std::int64_t rows, std::int64_t cols) {
  add_bias_kernel<<<blocks_for(rows * cols), kThreads, 0, stream_of(ctx)>>>(y, row, rows, cols);
}

void layer_norm(const Context& ctx, float* x, const float* w, const float* b, std::int64_t rows, std::int64_t cols,
                float eps) {
  norm_kernel<0><<<static_cast<unsigned>(rows), kThreads, 0, stream_of(ctx)>>>(x, w, b, cols, eps);
}

void rms_norm(const Context& ctx, float* x, const float* w, std::int64_t rows, std::int64_t cols, float eps) {
  norm_kernel<1><<<static_cast<unsigned>(rows), kThreads, 0, stream_of(ctx)>>>(x, w, nullptr, cols, eps);
}

void router_norm(const Context& ctx, float* x, const float* w, std::int64_t rows, std::int64_t cols, float eps) {
  norm_kernel<2><<<static_cast<unsigned>(rows), kThreads, 0, stream_of(ctx)>>>(x, w, nullptr, cols, eps);
}

void gelu_erf(const Context& ctx, float* x, std::int64_t n) {
  act_kernel<0><<<blocks_for(n), kThreads, 0, stream_of(ctx)>>>(x, n);
}

void quick_gelu(const Context& ctx, float* x, std::int64_t n) {
  act_kernel<1><<<blocks_for(n), kThreads, 0, stream_of(ctx)>>>(x, n);
}

void silu_mul(const Context& ctx, float* g, const float* u, std::int64_t n) {
  silu_mul_kernel<<<blocks_for(n), kThreads, 0, stream_of(ctx)>>>(g, u, n);
}

void softmax_rows(const Context& ctx, float* x, std::int64_t rows, std::int64_t cols) {
  softmax_kernel<<<static_cast<unsigned>(rows), kThreads, 0, stream_of(ctx)>>>(x, cols);
}

void scale_add_mask(const Context& ctx, float* x, float scale, const float* mask, std::int64_t batch, std::int64_t rows,
                    std::int64_t cols) {
  const std::int64_t total = batch * rows * cols;
  scale_mask_kernel<<<blocks_for(total), kThreads, 0, stream_of(ctx)>>>(x, scale, mask, rows * cols, total);
}

void rope(const Context& ctx, float* x, std::int64_t rows, std::int64_t heads, std::int64_t dim, const std::int64_t* tok,
          const std::int64_t* page, const float* inv_tok, std::int64_t n_tok, const float* inv_page, std::int64_t n_page,
          std::int64_t page_mod, std::int64_t one_tok, std::int64_t one_page) {
  (void)n_page;
  rope_kernel<<<blocks_for(rows * heads * (dim / 2)), kThreads, 0, stream_of(ctx)>>>(
      x, rows, heads, dim, tok, page, inv_tok, n_tok, inv_page, page_mod, one_tok, one_page);
}

void im2col(const Context& ctx, const float* x, std::int64_t c, std::int64_t h, std::int64_t w, std::int64_t kh,
            std::int64_t kw, int stride, int pad, float* cols) {
  const std::int64_t oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
  im2col_kernel<<<blocks_for(oh * ow * c * kh * kw), kThreads, 0, stream_of(ctx)>>>(x, c, h, w, kh, kw, stride, pad, oh,
                                                                                     ow, cols);
}

void chw_to_hwc(const Context& ctx, const float* x, std::int64_t c, std::int64_t h, std::int64_t w, float* y) {
  transpose_kernel<<<blocks_for(c * h * w), kThreads, 0, stream_of(ctx)>>>(x, c, h * w, y);
}

void hwc_to_chw(const Context& ctx, const float* x, std::int64_t h, std::int64_t w, std::int64_t c, float* y) {
  transpose_kernel<<<blocks_for(c * h * w), kThreads, 0, stream_of(ctx)>>>(x, h * w, c, y);
}

void gather_rows(const Context& ctx, const float* src, const std::int64_t* index, std::int64_t rows, std::int64_t cols,
                 float* dst) {
  gather_kernel<<<blocks_for(rows * cols), kThreads, 0, stream_of(ctx)>>>(src, index, rows, cols, dst);
}

void gather_rows_bf16(const Context& ctx, const std::uint16_t* table, const std::int64_t* index, std::int64_t rows,
                      std::int64_t cols, float* dst) {
  gather_bf16_kernel<<<blocks_for(rows * cols), kThreads, 0, stream_of(ctx)>>>(
      reinterpret_cast<const __nv_bfloat16*>(table), index, rows, cols, dst);
}

void scatter_add_rows(const Context& ctx, float* dst, const std::int64_t* index, const float* weight, const float* src,
                      std::int64_t rows, std::int64_t cols) {
  scatter_add_kernel<<<blocks_for(cols), kThreads, 0, stream_of(ctx)>>>(dst, index, weight, src, rows, cols);
}

void rel_pos_bias(const Context& ctx, const float* q, std::int64_t ldq, std::int64_t h, std::int64_t w, std::int64_t d,
                  const float* rh, const float* rw, float* bias) {
  const std::int64_t n = h * w;
  auto stream = stream_of(ctx);
  float* dots = nullptr;
  check(cudaMallocAsync(&dots, static_cast<std::size_t>(n * (h + w)) * 4, stream), "rel_pos scratch");
  rel_dot_kernel<<<blocks_for(n * (h + w)), kThreads, 0, stream>>>(q, ldq, h, w, d, rh, rw, dots);
  rel_expand_kernel<<<blocks_for(n * n), kThreads, 0, stream>>>(dots, h, w, bias);
  check(cudaFreeAsync(dots, stream), "rel_pos scratch free");
}

void block_summaries(const Context& ctx, const float* keys, std::int64_t s, std::int64_t d, const float* head_cls,
                     std::int64_t block_size, float* out) {
  const std::int64_t nblk = (s + block_size - 1) / block_size;
  block_summary_kernel<<<static_cast<unsigned>(nblk), 128, 0, stream_of(ctx)>>>(keys, s, d, head_cls, block_size, out);
}

}  // namespace uocr::cuda

namespace uocr::cuda {
namespace {

constexpr float kScoreMaskedDev = -1.0e9f;
constexpr float kEntmaxMaskedDev = -1.0e4f;
constexpr int kRouteThreads = 256;
constexpr int kMaxTop = 32;

// 1.5-entmax of `n` values already in descending order, in place; the same
// float operations as model/block_router.cpp's entmax15.
__device__ void entmax15_sorted(float* x, int n) {
  const float peak = x[0];
  for (int i = 0; i < n; ++i) x[i] = (x[i] - peak) / 2.0f;
  float sum = 0.0f, sum_sq = 0.0f;
  int support = 0;
  float tau[kMaxTop];
  for (int k = 0; k < n; ++k) {
    sum += x[k];
    sum_sq += x[k] * x[k];
    const float rho = static_cast<float>(k + 1);
    const float mean = sum / rho, mean_sq = sum_sq / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = fmaxf((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - sqrtf(delta);
    if (tau[k] <= x[k]) ++support;
  }
  const float tau_star = tau[max(support, 1) - 1];
  for (int i = 0; i < n; ++i) {
    const float t = fmaxf(x[i] - tau_star, 0.0f);
    x[i] = t * t;
  }
}

__global__ void route_kernel(const float* z, std::int64_t nblk, std::int64_t first_pos, std::int64_t keys,
                             const std::int64_t* block_min, const std::int64_t* block_max, std::int64_t prompt_blocks,
                             const std::int64_t* query_page, std::int64_t one_page, RouteParams p, float* mask,
                             const StepState* state, std::int64_t cap_blocks) {
  if (state != nullptr) {  // a decode step: the lengths come from the device
    first_pos = state->pos;
    keys = state->pos + 1;
    nblk = (keys + p.block_size - 1) / p.block_size;
    one_page = state->query_page;
  }
  extern __shared__ float shared[];
  float* x = shared;                                                                // nblk
  unsigned char* cand = reinterpret_cast<unsigned char*>(shared + cap_blocks);
  __shared__ float top_val[kMaxTop];
  __shared__ int top_idx[kMaxTop];
  __shared__ float sel_bias[kMaxTop];
  __shared__ int sel_block[kMaxTop];
  __shared__ int sel_valid[kMaxTop];
  __shared__ float red_val[kRouteThreads];
  __shared__ int red_idx[kRouteThreads];

  const std::int64_t row = blockIdx.x;
  const std::int64_t pos = first_pos + row;
  const std::int64_t last_remote = pos / p.block_size - p.local_blocks;
  const std::int64_t page = query_page != nullptr ? query_page[row] : one_page;
  const float* zr = z + row * nblk;

  int any_local = 0;
  for (std::int64_t b = threadIdx.x; b < nblk; b += blockDim.x) {
    const bool remote = b <= last_remote;
    const bool allowed = page >= 0 && b < prompt_blocks && block_min[b] >= 0 && block_min[b] <= page &&
                         page <= block_max[b];
    x[b] = remote ? fminf(fmaxf(zr[b], -1.0e4f), 1.0e4f) : kScoreMaskedDev;
    cand[b] = remote && allowed;
    any_local |= cand[b];
  }
  // A page with no image block in range means "no restriction" (the host's
  // empty candidate list), and so does a row with no candidate at all.
  const int any = __syncthreads_or(any_local);
  for (std::int64_t b = threadIdx.x; b < nblk; b += blockDim.x) {
    const bool remote = b <= last_remote;
    if (!any) cand[b] = remote;
    x[b] = cand[b] ? x[b] * p.idx_scale : kEntmaxMaskedDev;
  }
  __syncthreads();

  const int m = static_cast<int>(p.top_m < nblk ? (p.top_m > 1 ? p.top_m : 1) : (nblk > 1 ? nblk : 1));
  const int k = p.top_k < m ? (p.top_k > 1 ? static_cast<int>(p.top_k) : 1) : m;
  // Top m by repeated argmax; a tie takes the lower block index.
  for (int j = 0; j < m; ++j) {
    float best = -INFINITY;
    int best_i = INT_MAX;
    for (std::int64_t b = threadIdx.x; b < nblk; b += blockDim.x) {
      bool taken = false;
      for (int t = 0; t < j; ++t) taken |= top_idx[t] == b;
      if (taken) continue;
      if (x[b] > best || (x[b] == best && b < best_i)) {
        best = x[b];
        best_i = static_cast<int>(b);
      }
    }
    red_val[threadIdx.x] = best;
    red_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const float v = red_val[threadIdx.x + s];
        const int i = red_idx[threadIdx.x + s];
        if (v > red_val[threadIdx.x] || (v == red_val[threadIdx.x] && i < red_idx[threadIdx.x])) {
          red_val[threadIdx.x] = v;
          red_idx[threadIdx.x] = i;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      top_val[j] = red_val[0];
      top_idx[j] = red_idx[0];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    entmax15_sorted(top_val, m);
    float logp_sum = 0.0f;
    int count = 0;
    float logp[kMaxTop];
    for (int j = 0; j < k; ++j) {
      sel_block[j] = top_idx[j];
      sel_valid[j] = cand[top_idx[j]];
      logp[j] = logf(fmaxf(top_val[j], p.floor));
      if (sel_valid[j]) {
        logp_sum += logp[j];
        ++count;
      }
    }
    const float mu = logp_sum / static_cast<float>(max(count, 1));
    for (int j = 0; j < k; ++j) sel_bias[j] = sel_valid[j] ? (logp[j] - mu) / p.sigma : 0.0f;
  }
  __syncthreads();

  const std::int64_t window = p.local_blocks * p.block_size;
  float* out = mask + row * keys;
  for (std::int64_t j = threadIdx.x; j < keys; j += blockDim.x) {
    float v = -INFINITY;
    if (j <= pos) {
      if (j > pos - window) {
        v = 0.0f;
      } else {
        const int blk = static_cast<int>(j / p.block_size);
        for (int t = 0; t < k; ++t) {
          if (sel_valid[t] && sel_block[t] == blk) v = sel_bias[t];
        }
      }
    }
    out[j] = v;
  }
}

}  // namespace

void route_blocks(const Context& ctx, const float* z, std::int64_t rows, std::int64_t nblk, std::int64_t first_pos,
                  std::int64_t keys, const std::int64_t* block_min, const std::int64_t* block_max,
                  std::int64_t prompt_blocks, const std::int64_t* query_page, std::int64_t one_page,
                  const RouteParams& params, float* mask) {
  if (params.top_m > kMaxTop) throw std::invalid_argument("route_blocks: top_m above 32");
  const std::size_t shared = static_cast<std::size_t>(nblk) * 5;
  route_kernel<<<static_cast<unsigned>(rows), kRouteThreads, shared, stream_of(ctx)>>>(
      z, nblk, first_pos, keys, block_min, block_max, prompt_blocks, query_page, one_page, params, mask, nullptr, nblk);
}

void route_blocks_state(const Context& ctx, const float* z, std::int64_t max_blocks, const std::int64_t* block_min,
                        const std::int64_t* block_max, std::int64_t prompt_blocks, const StepState* state,
                        const RouteParams& params, float* mask) {
  if (params.top_m > kMaxTop) throw std::invalid_argument("route_blocks: top_m above 32");
  const std::size_t shared = static_cast<std::size_t>(max_blocks) * 5;
  route_kernel<<<1, kRouteThreads, shared, stream_of(ctx)>>>(z, 0, 0, 0, block_min, block_max, prompt_blocks, nullptr,
                                                             -1, params, mask, state, max_blocks);
}

}  // namespace uocr::cuda

namespace uocr::cuda {
namespace {

__global__ void gather_one_bf16_kernel(const __nv_bfloat16* table, std::int64_t index, std::int64_t cols, float* dst) {
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < cols) dst[i] = __bfloat162float(table[index * cols + i]);
}

// One warp per row: softmax, then top k by k warp-wide argmax rounds (lower
// index on a tie). Each lane holds experts lane, lane + 32, ...
constexpr int kTopkPerLane = 4;  // up to 128 experts

__global__ void expert_topk_kernel(const float* logits, std::int64_t rows, std::int64_t experts, std::int64_t k,
                                   std::int64_t* ids, float* weights) {
  const std::int64_t r = blockIdx.x * static_cast<std::int64_t>(blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (r >= rows) return;
  const float* row = logits + r * experts;
  float v[kTopkPerLane];
  float peak = -INFINITY;
  for (int i = 0; i < kTopkPerLane; ++i) {
    const std::int64_t e = lane + 32 * i;
    v[i] = e < experts ? row[e] : -INFINITY;
    peak = fmaxf(peak, v[i]);
  }
  for (int off = 16; off > 0; off >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffU, peak, off));
  double total = 0.0;
  for (int i = 0; i < kTopkPerLane; ++i) {
    if (lane + 32 * i < experts) total += expf(v[i] - peak);
  }
  for (int off = 16; off > 0; off >>= 1) total += __shfl_xor_sync(0xffffffffU, total, off);
  const float inv = static_cast<float>(1.0 / total);
  for (int i = 0; i < kTopkPerLane; ++i) v[i] = lane + 32 * i < experts ? expf(v[i] - peak) * inv : -INFINITY;
  for (std::int64_t j = 0; j < k; ++j) {
    float best = -INFINITY;
    int best_e = INT_MAX;
    for (int i = 0; i < kTopkPerLane; ++i) {
      const int e = lane + 32 * i;
      if (v[i] > best || (v[i] == best && e < best_e)) {
        best = v[i];
        best_e = e;
      }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffU, best, off);
      const int oe = __shfl_xor_sync(0xffffffffU, best_e, off);
      if (ob > best || (ob == best && oe < best_e)) {
        best = ob;
        best_e = oe;
      }
    }
    if (lane == 0) {
      ids[r * k + j] = best_e;
      weights[r * k + j] = best;
    }
    if (best_e % 32 == lane) v[best_e / 32] = -INFINITY;
  }
}

constexpr int kWarp = 32;

__device__ float warp_sum(float v) {
  for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffU, v, off);
  return v;
}

// Grid: (ffn rows / warps per block, k). One warp per output row of one
// expert slot; x is staged once per block in shared memory, bf16-rounded.
__global__ void expert_gate_up_kernel(const float* x, std::int64_t hidden, std::int64_t ffn,
                                      const __nv_bfloat16* gate, const __nv_bfloat16* up, const std::int64_t* ids,
                                      float* act) {
  extern __shared__ float xs[];
  for (std::int64_t i = threadIdx.x; i < hidden; i += blockDim.x) xs[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
  __syncthreads();
  const int warps = blockDim.x / kWarp, lane = threadIdx.x % kWarp;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + threadIdx.x / kWarp;
  const std::int64_t slot = blockIdx.y;
  if (row >= ffn) return;
  const std::int64_t e = ids[slot];
  const __nv_bfloat16* g = gate + (e * ffn + row) * hidden;
  const __nv_bfloat16* u = up + (e * ffn + row) * hidden;
  float sg = 0.0f, su = 0.0f;
  for (std::int64_t i = lane; i < hidden; i += kWarp) {
    sg += __bfloat162float(g[i]) * xs[i];
    su += __bfloat162float(u[i]) * xs[i];
  }
  sg = warp_sum(sg);
  su = warp_sum(su);
  if (lane == 0) {
    const float a = sg / (1.0f + expf(-sg)) * su;
    act[slot * ffn + row] = __bfloat162float(__float2bfloat16_rn(a));
  }
}

__global__ void expert_down_kernel(const float* act, std::int64_t hidden, std::int64_t ffn, const __nv_bfloat16* down,
                                   const std::int64_t* ids, const float* weights, std::int64_t k, float* out) {
  const int warps = blockDim.x / kWarp, lane = threadIdx.x % kWarp;
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + threadIdx.x / kWarp;
  if (row >= hidden) return;
  float total = 0.0f;
  for (std::int64_t j = 0; j < k; ++j) {
    const __nv_bfloat16* w = down + (ids[j] * hidden + row) * ffn;
    const float* a = act + j * ffn;
    float s = 0.0f;
    for (std::int64_t i = lane; i < ffn; i += kWarp) s += __bfloat162float(w[i]) * a[i];
    s = warp_sum(s);
    total += weights[j] * s;
  }
  if (lane == 0) out[row] = total;
}

}  // namespace

void gather_row_bf16(const Context& ctx, const std::uint16_t* table, std::int64_t one_index, std::int64_t cols,
                     float* dst) {
  gather_one_bf16_kernel<<<blocks_for(cols), kThreads, 0, stream_of(ctx)>>>(
      reinterpret_cast<const __nv_bfloat16*>(table), one_index, cols, dst);
}

void expert_topk(const Context& ctx, const float* logits, std::int64_t rows, std::int64_t experts, std::int64_t k,
                 std::int64_t* ids, float* weights) {
  if (experts > 128) throw std::invalid_argument("expert_topk: more than 128 experts");
  constexpr int kWarpsPerBlock = 8;
  expert_topk_kernel<<<static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock), kWarpsPerBlock * 32, 0,
                       stream_of(ctx)>>>(logits, rows, experts, k, ids, weights);
}

void expert_ffn_one(const Context& ctx, const float* x, std::int64_t hidden, std::int64_t ffn,
                    const std::uint16_t* gate, const std::uint16_t* up, const std::uint16_t* down,
                    const std::int64_t* ids, const float* weights, std::int64_t k, float* out) {
  auto stream = stream_of(ctx);
  float* act = nullptr;
  check(cudaMallocAsync(&act, static_cast<std::size_t>(k * ffn) * 4, stream), "expert scratch");
  constexpr int kBlock = 256, kWarps = kBlock / kWarp;
  const dim3 grid(static_cast<unsigned>((ffn + kWarps - 1) / kWarps), static_cast<unsigned>(k));
  expert_gate_up_kernel<<<grid, kBlock, static_cast<std::size_t>(hidden) * 4, stream>>>(
      x, hidden, ffn, reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up), ids, act);
  expert_down_kernel<<<static_cast<unsigned>((hidden + kWarps - 1) / kWarps), kBlock, 0, stream>>>(
      act, hidden, ffn, reinterpret_cast<const __nv_bfloat16*>(down), ids, weights, k, out);
  check(cudaFreeAsync(act, stream), "expert scratch free");
}

}  // namespace uocr::cuda
