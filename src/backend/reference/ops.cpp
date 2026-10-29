#include "backend/reference/ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "backend/reference/parallel.h"

namespace uocr::reference {
namespace {

std::atomic<unsigned> g_threads{std::max(1U, std::thread::hardware_concurrency())};

void require_shape(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(what);
  }
}

// Eight partial sums in a fixed order: vectorisable without fast-math, and
// the same result on every run and thread count.
float dot(const float* a, const float* b, std::int64_t n) {
  float acc[8] = {};
  std::int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (int lane = 0; lane < 8; ++lane) {
      acc[lane] += a[i + lane] * b[i + lane];
    }
  }
  float tail = 0.0f;
  for (; i < n; ++i) {
    tail += a[i] * b[i];
  }
  return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Keys cubic with a = -0.5, the antialiasing filter PyTorch uses.
float cubic_aa(float x) {
  constexpr float a = -0.5f;
  x = std::fabs(x);
  if (x < 1.0f) {
    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
  }
  if (x < 2.0f) {
    return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
  }
  return 0.0f;
}

struct AxisTaps {
  std::int64_t first;
  std::vector<float> weights;
};

std::vector<AxisTaps> aa_taps(std::int64_t in, std::int64_t out) {
  const float scale = static_cast<float>(in) / static_cast<float>(out);
  const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
  const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
  std::vector<AxisTaps> taps(static_cast<std::size_t>(out));
  for (std::int64_t i = 0; i < out; ++i) {
    const float center = scale * (static_cast<float>(i) + 0.5f);
    const auto lo = std::max<std::int64_t>(static_cast<std::int64_t>(center - support + 0.5f), 0);
    const auto hi = std::min<std::int64_t>(static_cast<std::int64_t>(center + support + 0.5f), in);
    auto& tap = taps[static_cast<std::size_t>(i)];
    tap.first = lo;
    float total = 0.0f;
    for (std::int64_t j = lo; j < hi; ++j) {
      const float w = cubic_aa((static_cast<float>(j) - center + 0.5f) * invscale);
      tap.weights.push_back(w);
      total += w;
    }
    if (total != 0.0f) {
      for (float& w : tap.weights) {
        w /= total;
      }
    }
  }
  return taps;
}

}  // namespace

unsigned thread_count() { return g_threads.load(std::memory_order_relaxed); }
void set_thread_count(unsigned threads) { g_threads.store(std::max(1U, threads), std::memory_order_relaxed); }

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
  require_shape(weight.rank() == 2 && x.shape().back() == weight.dim(1), "linear: x's last axis must match weight's K");
  require_shape(bias == nullptr || bias->numel() == weight.dim(0), "linear: bias must have N elements");
  const std::int64_t m = x.rows();
  const std::int64_t n = weight.dim(0);
  const std::int64_t k = weight.dim(1);
  auto shape = x.shape();
  shape.back() = n;
  Tensor out(std::move(shape));
  parallel_for(m, [&](std::int64_t r) {
    const float* xr = x.row(r);
    float* o = out.row(r);
    for (std::int64_t c = 0; c < n; ++c) {
      const float sum = dot(xr, weight.data() + c * k, k);
      o[c] = bias != nullptr ? sum + bias->data()[c] : sum;
    }
  });
  return out;
}

void layer_norm(Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
  const std::int64_t c = x.shape().back();
  require_shape(weight.numel() == c && bias.numel() == c, "layer_norm: weight and bias must match the last axis");
  parallel_for(x.rows(), [&](std::int64_t r) {
    float* v = x.row(r);
    double mean = 0.0;
    for (std::int64_t i = 0; i < c; ++i) mean += v[i];
    mean /= static_cast<double>(c);
    double var = 0.0;
    for (std::int64_t i = 0; i < c; ++i) var += (v[i] - mean) * (v[i] - mean);
    var /= static_cast<double>(c);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (std::int64_t i = 0; i < c; ++i) {
      v[i] = static_cast<float>((v[i] - mean) * inv) * weight.data()[i] + bias.data()[i];
    }
  });
}

void rms_norm(Tensor& x, const Tensor& weight, float eps) {
  const std::int64_t c = x.shape().back();
  require_shape(weight.numel() == c, "rms_norm: weight must match the last axis");
  parallel_for(x.rows(), [&](std::int64_t r) {
    float* v = x.row(r);
    double ss = 0.0;
    for (std::int64_t i = 0; i < c; ++i) ss += static_cast<double>(v[i]) * v[i];
    const auto inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(c) + eps));
    for (std::int64_t i = 0; i < c; ++i) {
      v[i] = weight.data()[i] * (v[i] * inv);
    }
  });
}

void layer_norm_channels(Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
  require_shape(x.rank() == 3 && weight.numel() == x.dim(0) && bias.numel() == x.dim(0),
                "layer_norm_channels: x must be [C, H, W] with C-sized weight and bias");
  Tensor hwc = chw_to_hwc(x);
  layer_norm(hwc, weight, bias, eps);
  x = hwc_to_chw(hwc);
}

void gelu_erf(Tensor& x) {
  for (float& v : x.span()) {
    v = 0.5f * v * (1.0f + std::erf(v * static_cast<float>(M_SQRT1_2)));
  }
}

void quick_gelu(Tensor& x) {
  for (float& v : x.span()) {
    v = v * sigmoid(1.702f * v);
  }
}

void silu(Tensor& x) {
  for (float& v : x.span()) {
    v = v * sigmoid(v);
  }
}

void softmax_rows(Tensor& x) {
  const std::int64_t c = x.shape().back();
  parallel_for(x.rows(), [&](std::int64_t r) {
    float* v = x.row(r);
    float peak = -std::numeric_limits<float>::infinity();
    for (std::int64_t i = 0; i < c; ++i) peak = std::max(peak, v[i]);
    if (peak == -std::numeric_limits<float>::infinity()) {
      std::fill(v, v + c, 0.0f);  // a fully masked row attends to nothing
      return;
    }
    double total = 0.0;
    for (std::int64_t i = 0; i < c; ++i) {
      v[i] = std::exp(v[i] - peak);
      total += v[i];
    }
    const auto inv = static_cast<float>(1.0 / total);
    for (std::int64_t i = 0; i < c; ++i) v[i] *= inv;
  });
}

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias, int stride, int padding) {
  require_shape(x.rank() == 3 && weight.rank() == 4 && weight.dim(1) == x.dim(0),
                "conv2d: x must be [C, H, W] and weight [O, C, KH, KW]");
  const std::int64_t c = x.dim(0), h = x.dim(1), w = x.dim(2);
  const std::int64_t o = weight.dim(0), kh = weight.dim(2), kw = weight.dim(3);
  const std::int64_t oh = (h + 2 * padding - kh) / stride + 1;
  const std::int64_t ow = (w + 2 * padding - kw) / stride + 1;
  // im2col: one row per output position, laid out [C, KH, KW] like the weight.
  const std::int64_t patch = c * kh * kw;
  Tensor cols({oh * ow, patch});
  parallel_for(oh, [&](std::int64_t oy) {
    for (std::int64_t ox = 0; ox < ow; ++ox) {
      float* col = cols.row(oy * ow + ox);
      for (std::int64_t ci = 0; ci < c; ++ci) {
        for (std::int64_t ky = 0; ky < kh; ++ky) {
          const std::int64_t iy = oy * stride - padding + ky;
          for (std::int64_t kx = 0; kx < kw; ++kx) {
            const std::int64_t ix = ox * stride - padding + kx;
            const bool inside = iy >= 0 && iy < h && ix >= 0 && ix < w;
            *col++ = inside ? x.data()[(ci * h + iy) * w + ix] : 0.0f;
          }
        }
      }
    }
  });
  const Tensor flat_weight({o, patch}, std::vector<float>(weight.span().begin(), weight.span().end()));
  Tensor hwc = linear(cols, flat_weight, bias).reshaped({oh, ow, o});
  return hwc_to_chw(hwc);
}

Tensor attention_head(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* bias, float scale) {
  require_shape(q.rank() == 2 && k.rank() == 2 && v.rank() == 2 && q.dim(1) == k.dim(1) && k.dim(0) == v.dim(0),
                "attention_head: q [Lq, D], k [Lk, D], v [Lk, Dv]");
  const std::int64_t lq = q.dim(0), lk = k.dim(0), d = q.dim(1), dv = v.dim(1);
  require_shape(bias == nullptr || (bias->dim(0) == lq && bias->dim(1) == lk), "attention_head: bias must be [Lq, Lk]");
  Tensor out({lq, dv});
  parallel_for(lq, [&](std::int64_t i) {
    std::vector<float> scores(static_cast<std::size_t>(lk));
    float peak = -std::numeric_limits<float>::infinity();
    for (std::int64_t j = 0; j < lk; ++j) {
      float s = dot(q.row(i), k.row(j), d) * scale;
      if (bias != nullptr) s += bias->row(i)[j];
      scores[static_cast<std::size_t>(j)] = s;
      peak = std::max(peak, s);
    }
    float* o = out.row(i);
    if (peak == -std::numeric_limits<float>::infinity()) {
      return;  // every key masked: zeros
    }
    double total = 0.0;
    for (float& s : scores) {
      s = std::exp(s - peak);
      total += s;
    }
    const auto inv = static_cast<float>(1.0 / total);
    for (std::int64_t j = 0; j < lk; ++j) {
      const float p = scores[static_cast<std::size_t>(j)] * inv;
      const float* vr = v.row(j);
      for (std::int64_t e = 0; e < dv; ++e) o[e] += p * vr[e];
    }
  });
  return out;
}

Tensor resize_bicubic_aa(const Tensor& x, std::int64_t out_h, std::int64_t out_w) {
  require_shape(x.rank() == 3, "resize_bicubic_aa: x must be [C, H, W]");
  const std::int64_t c = x.dim(0), h = x.dim(1), w = x.dim(2);
  const auto rows = aa_taps(h, out_h);
  const auto cols = aa_taps(w, out_w);
  Tensor out({c, out_h, out_w});
  parallel_for(c, [&](std::int64_t ch) {
    const float* src = x.data() + ch * h * w;
    float* dst = out.data() + ch * out_h * out_w;
    for (std::int64_t oy = 0; oy < out_h; ++oy) {
      const auto& ry = rows[static_cast<std::size_t>(oy)];
      for (std::int64_t ox = 0; ox < out_w; ++ox) {
        const auto& rx = cols[static_cast<std::size_t>(ox)];
        float acc = 0.0f;
        for (std::size_t a = 0; a < ry.weights.size(); ++a) {
          const float* line = src + (ry.first + static_cast<std::int64_t>(a)) * w + rx.first;
          float line_acc = 0.0f;
          for (std::size_t b = 0; b < rx.weights.size(); ++b) line_acc += line[b] * rx.weights[b];
          acc += line_acc * ry.weights[a];
        }
        dst[oy * out_w + ox] = acc;
      }
    }
  });
  return out;
}

Tensor resize_linear_rows(const Tensor& x, std::int64_t out_len) {
  require_shape(x.rank() == 2, "resize_linear_rows: x must be [L, C]");
  const std::int64_t in = x.dim(0), c = x.dim(1);
  const float scale = static_cast<float>(in) / static_cast<float>(out_len);
  Tensor out({out_len, c});
  for (std::int64_t i = 0; i < out_len; ++i) {
    const float src = std::max(scale * (static_cast<float>(i) + 0.5f) - 0.5f, 0.0f);
    const auto i0 = static_cast<std::int64_t>(src);
    const std::int64_t i1 = i0 + (i0 < in - 1 ? 1 : 0);
    const float l1 = src - static_cast<float>(i0);
    const float l0 = 1.0f - l1;
    for (std::int64_t e = 0; e < c; ++e) {
      out.row(i)[e] = l0 * x.row(i0)[e] + l1 * x.row(i1)[e];
    }
  }
  return out;
}

Tensor chw_to_hwc(const Tensor& x) {
  const std::int64_t c = x.dim(0), h = x.dim(1), w = x.dim(2);
  Tensor out({h, w, c});
  for (std::int64_t ch = 0; ch < c; ++ch)
    for (std::int64_t p = 0; p < h * w; ++p) out.data()[p * c + ch] = x.data()[ch * h * w + p];
  return out;
}

Tensor hwc_to_chw(const Tensor& x) {
  const std::int64_t h = x.dim(0), w = x.dim(1), c = x.dim(2);
  Tensor out({c, h, w});
  for (std::int64_t p = 0; p < h * w; ++p)
    for (std::int64_t ch = 0; ch < c; ++ch) out.data()[ch * h * w + p] = x.data()[p * c + ch];
  return out;
}

}  // namespace uocr::reference
