#include "model/clip_encoder.h"

#include <cmath>
#include <format>
#include <stdexcept>

#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"

namespace uocr {
namespace ref = reference;
namespace {

constexpr float kNormEps = 1e-5f;

void add_into(Tensor& x, const Tensor& y) {
  for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] += y.data()[i];
}

}  // namespace

ClipEncoder::ClipEncoder(const WeightSource& weights, const ClipDims& dims) : dims_(dims) {
  const std::string root = "model.vision_model";
  class_embedding_ = weights.get(root + ".embeddings.class_embedding");
  position_embedding_ = weights.get(root + ".embeddings.position_embedding.weight");
  pre_ln_w_ = weights.get(root + ".pre_layrnorm.weight");
  pre_ln_b_ = weights.get(root + ".pre_layrnorm.bias");
  for (std::int64_t i = 0; i < dims.layers; ++i) {
    const std::string p = std::format("{}.transformer.layers.{}.", root, i);
    layers_.push_back(Layer{
        weights.get(p + "layer_norm1.weight"), weights.get(p + "layer_norm1.bias"),
        weights.get(p + "self_attn.qkv_proj.weight"), weights.get(p + "self_attn.qkv_proj.bias"),
        weights.get(p + "self_attn.out_proj.weight"), weights.get(p + "self_attn.out_proj.bias"),
        weights.get(p + "layer_norm2.weight"), weights.get(p + "layer_norm2.bias"),
        weights.get(p + "mlp.fc1.weight"), weights.get(p + "mlp.fc1.bias"),
        weights.get(p + "mlp.fc2.weight"), weights.get(p + "mlp.fc2.bias")});
  }
}

// The learned 16 x 16 grid of positions (plus the class position), resized
// to `grid` x `grid` when the features come from a smaller view.
Tensor ClipEncoder::positions(std::int64_t grid) const {
  const std::int64_t width = position_embedding_.dim(1);
  const auto src = static_cast<std::int64_t>(std::sqrt(static_cast<double>(position_embedding_.dim(0) - 1)));
  if (src == grid) {
    return position_embedding_;
  }
  Tensor old_grid({src, src, width}, std::vector<float>(position_embedding_.data() + width,
                                                        position_embedding_.data() + position_embedding_.numel()));
  const Tensor resized = ref::chw_to_hwc(ref::resize_bicubic_aa(ref::hwc_to_chw(old_grid), grid, grid));
  Tensor out({1 + grid * grid, width});
  std::copy_n(position_embedding_.data(), width, out.data());
  std::copy_n(resized.data(), resized.numel(), out.data() + width);
  return out;
}

Tensor ClipEncoder::forward(const Tensor& sam_features) const {
  if (sam_features.rank() != 3 || sam_features.dim(0) != dims_.width || sam_features.dim(1) != sam_features.dim(2)) {
    throw std::invalid_argument("ClipEncoder: features must be [1024, g, g]");
  }
  const std::int64_t g = sam_features.dim(1), n = 1 + g * g, c = dims_.width;
  const Tensor cells = ref::chw_to_hwc(sam_features);  // [g, g, C] in row order
  Tensor x({n, c});
  std::copy_n(class_embedding_.data(), c, x.data());
  std::copy_n(cells.data(), cells.numel(), x.data() + c);
  add_into(x, positions(g));
  ref::layer_norm(x, pre_ln_w_, pre_ln_b_, kNormEps);

  const std::int64_t heads = dims_.heads, hd = c / heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  for (const auto& layer : layers_) {
    Tensor normed = x;
    ref::layer_norm(normed, layer.ln1_w, layer.ln1_b, kNormEps);
    const Tensor qkv = ref::linear(normed, layer.qkv_w, &layer.qkv_b);  // [n, 3 * C]: (3, heads, hd)
    Tensor merged({n, c});
    ref::parallel_for(heads, [&](std::int64_t head) {
      Tensor q({n, hd}), k({n, hd}), v({n, hd});
      for (std::int64_t t = 0; t < n; ++t) {
        const float* r = qkv.row(t);
        std::copy_n(r + head * hd, hd, q.row(t));
        std::copy_n(r + c + head * hd, hd, k.row(t));
        std::copy_n(r + 2 * c + head * hd, hd, v.row(t));
      }
      const Tensor out = ref::attention_head(q, k, v, nullptr, scale);
      for (std::int64_t t = 0; t < n; ++t) std::copy_n(out.row(t), hd, merged.row(t) + head * hd);
    });
    add_into(x, ref::linear(merged, layer.out_w, &layer.out_b));

    Tensor hidden = x;
    ref::layer_norm(hidden, layer.ln2_w, layer.ln2_b, kNormEps);
    hidden = ref::linear(hidden, layer.fc1_w, &layer.fc1_b);
    ref::quick_gelu(hidden);
    add_into(x, ref::linear(hidden, layer.fc2_w, &layer.fc2_b));
  }
  return x;
}

}  // namespace uocr
