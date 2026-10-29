#include "model/sam_encoder.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"

namespace uocr {
namespace ref = reference;
namespace {

constexpr float kNormEps = 1e-6f;

// x [H, W, C] zero-padded to multiples of `win`, split into windows in row
// order: [n_windows, win * win, C].
Tensor partition(const Tensor& x, std::int64_t win, std::int64_t& hp, std::int64_t& wp) {
  const std::int64_t h = x.dim(0), w = x.dim(1), c = x.dim(2);
  hp = (h + win - 1) / win * win;
  wp = (w + win - 1) / win * win;
  const std::int64_t nh = hp / win, nw = wp / win;
  Tensor out({nh * nw, win * win, c});
  for (std::int64_t y = 0; y < h; ++y) {
    for (std::int64_t xx = 0; xx < w; ++xx) {
      const std::int64_t window = (y / win) * nw + xx / win;
      const std::int64_t slot = (y % win) * win + xx % win;
      std::copy_n(x.data() + (y * w + xx) * c, c, out.data() + (window * win * win + slot) * c);
    }
  }
  return out;
}

Tensor unpartition(const Tensor& windows, std::int64_t win, std::int64_t hp, std::int64_t wp, std::int64_t h,
                   std::int64_t w) {
  const std::int64_t c = windows.dim(2), nw = wp / win;
  (void)hp;
  Tensor out({h, w, c});
  for (std::int64_t y = 0; y < h; ++y) {
    for (std::int64_t xx = 0; xx < w; ++xx) {
      const std::int64_t window = (y / win) * nw + xx / win;
      const std::int64_t slot = (y % win) * win + xx % win;
      std::copy_n(windows.data() + (window * win * win + slot) * c, c, out.data() + (y * w + xx) * c);
    }
  }
  return out;
}

}  // namespace

Tensor relative_positions(const Tensor& table, std::int64_t size) {
  const std::int64_t span = 2 * size - 1;
  const Tensor resized = table.dim(0) == span ? table : ref::resize_linear_rows(table, span);
  const std::int64_t d = resized.dim(1);
  Tensor out({size, size, d});
  for (std::int64_t q = 0; q < size; ++q) {
    for (std::int64_t k = 0; k < size; ++k) {
      std::copy_n(resized.row(q - k + size - 1), d, out.data() + (q * size + k) * d);
    }
  }
  return out;
}

SamEncoder::SamEncoder(const WeightSource& weights, const SamDims& dims) : dims_(dims) {
  const std::string root = "model.sam_model";
  patch_w_ = weights.get(root + ".patch_embed.proj.weight");
  patch_b_ = weights.get(root + ".patch_embed.proj.bias");
  pos_embed_ = weights.get(root + ".pos_embed");
  for (std::int64_t i = 0; i < dims.layers; ++i) {
    const std::string p = std::format("{}.blocks.{}.", root, i);
    Block b;
    b.norm1_w = weights.get(p + "norm1.weight");
    b.norm1_b = weights.get(p + "norm1.bias");
    b.qkv_w = weights.get(p + "attn.qkv.weight");
    b.qkv_b = weights.get(p + "attn.qkv.bias");
    b.proj_w = weights.get(p + "attn.proj.weight");
    b.proj_b = weights.get(p + "attn.proj.bias");
    b.rel_pos_h = weights.get(p + "attn.rel_pos_h");
    b.rel_pos_w = weights.get(p + "attn.rel_pos_w");
    b.norm2_w = weights.get(p + "norm2.weight");
    b.norm2_b = weights.get(p + "norm2.bias");
    b.lin1_w = weights.get(p + "mlp.lin1.weight");
    b.lin1_b = weights.get(p + "mlp.lin1.bias");
    b.lin2_w = weights.get(p + "mlp.lin2.weight");
    b.lin2_b = weights.get(p + "mlp.lin2.bias");
    b.global = std::ranges::contains(dims.global_layers, i);
    blocks_.push_back(std::move(b));
  }
  neck0_w_ = weights.get(root + ".neck.0.weight");
  neck1_w_ = weights.get(root + ".neck.1.weight");
  neck1_b_ = weights.get(root + ".neck.1.bias");
  neck2_w_ = weights.get(root + ".neck.2.weight");
  neck3_w_ = weights.get(root + ".neck.3.weight");
  neck3_b_ = weights.get(root + ".neck.3.bias");
  net2_w_ = weights.get(root + ".net_2.weight");
  net3_w_ = weights.get(root + ".net_3.weight");
}

Tensor SamEncoder::forward(const Tensor& image) const {
  if (image.rank() != 3 || image.dim(0) != 3 || image.dim(1) % 64 != 0 || image.dim(1) != image.dim(2)) {
    throw std::invalid_argument("SamEncoder: image must be [3, S, S] with S a multiple of 64");
  }
  Tensor x = ref::chw_to_hwc(ref::conv2d(image, patch_w_, &patch_b_, static_cast<int>(dims_.patch), 0));
  const std::int64_t g = x.dim(0);
  // pos_embed is [1, 64, 64, C]; other grids get it resized like the model does.
  Tensor pos = Tensor(std::vector<std::int64_t>(pos_embed_.shape().begin() + 1, pos_embed_.shape().end()),
                      std::vector<float>(pos_embed_.span().begin(), pos_embed_.span().end()));
  if (pos.dim(0) != g) {
    pos = ref::chw_to_hwc(ref::resize_bicubic_aa(ref::hwc_to_chw(pos), g, g));
  }
  for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] += pos.data()[i];

  for (const auto& block : blocks_) {
    x = block_forward(block, std::move(x));
  }

  Tensor y = ref::conv2d(ref::hwc_to_chw(x), neck0_w_, nullptr, 1, 0);
  ref::layer_norm_channels(y, neck1_w_, neck1_b_, kNormEps);
  y = ref::conv2d(y, neck2_w_, nullptr, 1, 1);
  ref::layer_norm_channels(y, neck3_w_, neck3_b_, kNormEps);
  y = ref::conv2d(y, net2_w_, nullptr, 2, 1);
  return ref::conv2d(y, net3_w_, nullptr, 2, 1);
}

Tensor SamEncoder::block_forward(const Block& block, Tensor x) const {
  const std::int64_t h = x.dim(0), w = x.dim(1), c = x.dim(2);
  Tensor normed = x;
  ref::layer_norm(normed, block.norm1_w, block.norm1_b, kNormEps);
  Tensor attended;
  if (block.global) {
    attended = attention(block, std::move(normed).reshaped({h * w, c}), h, w).reshaped({h, w, c});
  } else {
    std::int64_t hp = 0, wp = 0;
    const Tensor windows = partition(normed, dims_.window, hp, wp);
    const std::int64_t n = dims_.window * dims_.window;
    Tensor out(windows.shape());
    for (std::int64_t i = 0; i < windows.dim(0); ++i) {
      const Tensor win = Tensor({n, c}, std::vector<float>(windows.data() + i * n * c, windows.data() + (i + 1) * n * c));
      const Tensor result = attention(block, win, dims_.window, dims_.window);
      std::copy_n(result.data(), n * c, out.data() + i * n * c);
    }
    attended = unpartition(out, dims_.window, hp, wp, h, w);
  }
  for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] += attended.data()[i];

  Tensor hidden = x;
  ref::layer_norm(hidden, block.norm2_w, block.norm2_b, kNormEps);
  hidden = ref::linear(hidden, block.lin1_w, &block.lin1_b);
  ref::gelu_erf(hidden);
  hidden = ref::linear(hidden, block.lin2_w, &block.lin2_b);
  for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] += hidden.data()[i];
  return x;
}

// Multi-head attention over one window of h x w tokens, x [h * w, C], with
// SAM's decomposed relative-position bias computed from the unscaled query.
Tensor SamEncoder::attention(const Block& block, const Tensor& x, std::int64_t h, std::int64_t w) const {
  const std::int64_t n = h * w, c = x.dim(1), heads = dims_.heads, hd = c / heads;
  const Tensor qkv = ref::linear(x, block.qkv_w, &block.qkv_b);  // [n, 3 * C]: (3, heads, hd)
  const Tensor rh = relative_positions(block.rel_pos_h, h);      // [h, h, hd]
  const Tensor rw = relative_positions(block.rel_pos_w, w);      // [w, w, hd]
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  Tensor merged({n, c});
  ref::parallel_for(heads, [&](std::int64_t head) {
    Tensor q({n, hd}), k({n, hd}), v({n, hd});
    for (std::int64_t t = 0; t < n; ++t) {
      const float* r = qkv.row(t);
      std::copy_n(r + head * hd, hd, q.row(t));
      std::copy_n(r + c + head * hd, hd, k.row(t));
      std::copy_n(r + 2 * c + head * hd, hd, v.row(t));
    }
    Tensor bias({n, n});
    for (std::int64_t t = 0; t < n; ++t) {
      const std::int64_t qy = t / w, qx = t % w;
      const float* qt = q.row(t);
      std::vector<float> by(static_cast<std::size_t>(h)), bx(static_cast<std::size_t>(w));
      for (std::int64_t ky = 0; ky < h; ++ky) {
        const float* r = rh.data() + (qy * h + ky) * hd;
        float s = 0.0f;
        for (std::int64_t e = 0; e < hd; ++e) s += qt[e] * r[e];
        by[static_cast<std::size_t>(ky)] = s;
      }
      for (std::int64_t kx = 0; kx < w; ++kx) {
        const float* r = rw.data() + (qx * w + kx) * hd;
        float s = 0.0f;
        for (std::int64_t e = 0; e < hd; ++e) s += qt[e] * r[e];
        bx[static_cast<std::size_t>(kx)] = s;
      }
      float* br = bias.row(t);
      for (std::int64_t ky = 0; ky < h; ++ky)
        for (std::int64_t kx = 0; kx < w; ++kx) br[ky * w + kx] = by[static_cast<std::size_t>(ky)] + bx[static_cast<std::size_t>(kx)];
    }
    const Tensor out = ref::attention_head(q, k, v, &bias, scale);
    for (std::int64_t t = 0; t < n; ++t) std::copy_n(out.row(t), hd, merged.row(t) + head * hd);
  });
  return ref::linear(merged, block.proj_w, &block.proj_b);
}

}  // namespace uocr
