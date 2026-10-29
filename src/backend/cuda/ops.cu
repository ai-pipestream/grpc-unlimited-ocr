#include "backend/cuda/ops.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>

#include "backend/cuda/context_impl.h"

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
                            std::int64_t page_mod) {
  const std::int64_t half = dim / 2;
  const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * heads * half) return;
  const std::int64_t r = i / (heads * half), h = (i / half) % heads, p = i % half;
  std::int64_t pg = page[r];
  if (page_mod > 0) pg = ((pg % page_mod) + page_mod) % page_mod;
  const float angle = p < n_tok ? static_cast<float>(tok[r]) * inv_tok[p] : static_cast<float>(pg) * inv_page[p - n_tok];
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

}  // namespace

void linear(const Context& ctx, const float* x, std::int64_t m, std::int64_t k, const std::uint16_t* w_bf16,
            std::int64_t n, const float* bias, float* y) {
  auto stream = stream_of(ctx);
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
          std::int64_t page_mod) {
  (void)n_page;
  rope_kernel<<<blocks_for(rows * heads * (dim / 2)), kThreads, 0, stream_of(ctx)>>>(x, rows, heads, dim, tok, page,
                                                                                      inv_tok, n_tok, inv_page, page_mod);
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
