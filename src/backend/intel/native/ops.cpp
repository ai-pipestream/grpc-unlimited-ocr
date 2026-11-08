#include "backend/intel/native/ops.h"

#include <cmath>

#include "backend/intel/native/context_impl.h"
namespace uocr::intel_native {
void copy(const Context& c, float* dst, const float* src, std::int64_t n) {
  c.impl().queue.memcpy(dst, src, static_cast<std::size_t>(n) * 4);
}
void add_inplace(const Context& c, float* y, const float* x, std::int64_t n) {
  each(c, n, [=](auto i) { y[i] += x[i]; });
}
void add_rows(const Context& c, float* y, const float* x, std::int64_t r, std::int64_t n) {
  each(c, r * n, [=](auto i) { y[i] += x[i % n]; });
}
void gelu_erf(const Context& c, float* x, std::int64_t n) {
  each(c, n, [=](auto i) { x[i] = 0.5f * x[i] * (1 + sycl::erf(x[i] * 0.70710678118654752f)); });
}
void quick_gelu(const Context& c, float* x, std::int64_t n) {
  each(c, n, [=](auto i) { x[i] /= 1 + sycl::exp(-1.702f * x[i]); });
}
void silu_mul(const Context& c, float* x, const float* u, std::int64_t n) {
  each(c, n, [=](auto i) { x[i] = x[i] / (1 + sycl::exp(-x[i])) * u[i]; });
}
namespace {
void norm(const Context& c, float* x, const float* w, std::int64_t rows, std::int64_t cols, float eps, bool router) {
  each(c, rows, [=](auto r) {
    float* p = x + r * cols;
    float sq = 0;
    for (std::int64_t j = 0; j < cols; ++j) {
      if (router && !sycl::isfinite(p[j])) p[j] = 0;
      sq += p[j] * p[j];
    }
    const float inv = sycl::rsqrt(sycl::fmax(sq / static_cast<float>(cols), 0.0f) + eps);
    for (std::int64_t j = 0; j < cols; ++j) {
      float v = p[j] * inv;
      if (router) v = sycl::clamp(v, -1e4f, 1e4f);
      p[j] = router ? sycl::clamp(v * (1 + w[j]), -1e4f, 1e4f) : v * w[j];
    }
  });
}
}  // namespace
void rms_norm(const Context& c, float* x, const float* w, std::int64_t r, std::int64_t n, float e) {
  norm(c, x, w, r, n, e, false);
}
void router_norm(const Context& c, float* x, const float* w, std::int64_t r, std::int64_t n, float e) {
  norm(c, x, w, r, n, e, true);
}
void softmax_rows(const Context& c, float* x, std::int64_t rows, std::int64_t cols) {
  each(c, rows, [=](auto r) {
    float* p = x + r * cols;
    float peak = -INFINITY, total = 0;
    for (std::int64_t j = 0; j < cols; ++j) peak = sycl::fmax(peak, p[j]);
    if (peak == -INFINITY) {
      for (std::int64_t j = 0; j < cols; ++j) p[j] = 0;
      return;
    }
    for (std::int64_t j = 0; j < cols; ++j) {
      p[j] = sycl::exp(p[j] - peak);
      total += p[j];
    }
    for (std::int64_t j = 0; j < cols; ++j) p[j] /= total;
  });
}
void scale_add_mask(const Context& c, float* x, float scale, const float* mask, std::int64_t batch, std::int64_t r,
                    std::int64_t n) {
  each(c, batch * r * n, [=](auto i) { x[i] = x[i] * scale + (mask ? mask[i % (r * n)] : 0); });
}
void rope(const Context& c, float* x, std::int64_t rows, std::int64_t heads, std::int64_t dim, const std::int64_t* tok,
          const std::int64_t* page, const float* it, std::int64_t nt, const float* ip, std::int64_t np,
          std::int64_t mod, std::int64_t ot, std::int64_t op) {
  if (nt + np != dim / 2) throw std::invalid_argument("rope dimensions");
  each(c, rows * heads * (dim / 2), [=](auto i) {
    const auto half = dim / 2, r = i / (heads * half), h = (i / half) % heads, p = i % half;
    auto pg = page ? page[r] : op;
    const auto tk = tok ? tok[r] : ot;
    if (mod > 0) pg = ((pg % mod) + mod) % mod;
    const float angle = p < nt ? static_cast<float>(tk) * it[p] : static_cast<float>(pg) * ip[p - nt];
    const float s = sycl::sin(angle), cs = sycl::cos(angle);
    float* v = x + r * heads * dim + h * dim;
    const float a = v[p], b = v[p + half];
    v[p] = a * cs - b * s;
    v[p + half] = b * cs + a * s;
  });
}
void im2col(const Context& c, const float* x, std::int64_t channels, std::int64_t h, std::int64_t w, std::int64_t kh,
            std::int64_t kw, int stride, int pad, float* out) {
  const auto oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, patch = channels * kh * kw;
  each(c, oh * ow * patch, [=](auto i) {
    const auto pos = i / patch, e = i % patch, ci = e / (kh * kw), ky = (e / kw) % kh, kx = e % kw;
    const auto iy = (pos / ow) * stride - pad + ky, ix = (pos % ow) * stride - pad + kx;
    out[i] = (iy >= 0 && iy < h && ix >= 0 && ix < w) ? x[(ci * h + iy) * w + ix] : 0;
  });
}
void chw_to_hwc(const Context& c, const float* x, std::int64_t ch, std::int64_t h, std::int64_t w, float* y) {
  each(c, ch * h * w, [=](auto i) { y[(i % (h * w)) * ch + i / (h * w)] = x[i]; });
}
void hwc_to_chw(const Context& c, const float* x, std::int64_t h, std::int64_t w, std::int64_t ch, float* y) {
  each(c, ch * h * w, [=](auto i) { y[(i % ch) * (h * w) + i / ch] = x[i]; });
}
void gather_rows(const Context& c, const float* src, const std::int64_t* ids, std::int64_t r, std::int64_t n,
                 float* dst) {
  each(c, r * n, [=](auto i) { dst[i] = src[ids[i / n] * n + i % n]; });
}
void gather_rows_bf16(const Context& c, const std::uint16_t* src, const std::int64_t* ids, std::int64_t r,
                      std::int64_t n, float* dst) {
  each(c, r * n, [=](auto i) { dst[i] = wide(src[ids[i / n] * n + i % n]); });
}
void gather_row_bf16(const Context& c, const std::uint16_t* src, std::int64_t id, std::int64_t n, float* dst) {
  each(c, n, [=](auto i) { dst[i] = wide(src[id * n + i]); });
}
void scatter_add_rows(const Context& c, float* dst, const std::int64_t* ids, const float* weights, const float* src,
                      std::int64_t rows, std::int64_t cols) {
  each(c, cols, [=](auto j) {
    for (std::int64_t i = 0; i < rows; ++i) dst[ids[i] * cols + j] += weights[i] * src[i * cols + j];
  });
}
void concat_cols(const Context& c, const float* a, std::int64_t ca, const float* b, std::int64_t cb, std::int64_t rows,
                 float* out) {
  each(c, rows * (ca + cb), [=](auto i) {
    const auto r = i / (ca + cb), j = i % (ca + cb);
    out[i] = j < ca ? a[r * ca + j] : b[r * cb + j - ca];
  });
}
void rel_pos_bias(const Context& c, const float* q, std::int64_t ldq, std::int64_t h, std::int64_t w, std::int64_t d,
                  const float* rh, const float* rw, float* bias) {
  const auto n = h * w, span = h + w;
  Array temp(c, DType::kF32, n * span);
  float* dots = temp.f32();
  each(c, n * span, [=](auto i) {
    const auto t = i / span, j = i % span;
    const float* r = j < h ? rh + ((t / w) * h + j) * d : rw + ((t % w) * w + j - h) * d;
    float sum = 0;
    for (std::int64_t e = 0; e < d; ++e) sum += q[t * ldq + e] * r[e];
    dots[i] = sum;
  });
  each(c, n * n, [=](auto i) { bias[i] = dots[(i / n) * span + (i % n) / w] + dots[(i / n) * span + h + i % w]; });
}
}  // namespace uocr::intel_native
