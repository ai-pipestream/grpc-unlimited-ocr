#include "model/cuda/cuda_vision.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "backend/cuda/ops.h"
#include "backend/reference/ops.h"
#include "model/sam_encoder.h"

namespace uocr::cuda {
namespace {

constexpr float kSamEps = 1e-6f;
constexpr float kClipEps = 1e-5f;

Array f32(const Context& ctx, std::int64_t n) { return Array(ctx, DType::kF32, n); }

void apply(const Context& ctx, const Array& x, std::int64_t rows, const DeviceMatrix& w, const Array* bias, Array& y) {
  linear(ctx, x.f32(), rows, w.cols, w.data.bf16(), w.rows, bias != nullptr ? bias->f32() : nullptr, y.f32());
}

// Window layout for a g x g grid: for each window slot the source row (or
// the zero row g * g for padding), and for each grid cell its window row.
struct WindowIndex {
  std::vector<std::int64_t> gather, scatter;
  std::int64_t windows = 0;
};

WindowIndex window_index(std::int64_t g, std::int64_t win) {
  const std::int64_t nw = (g + win - 1) / win;
  WindowIndex idx;
  idx.windows = nw * nw;
  idx.gather.assign(static_cast<std::size_t>(idx.windows * win * win), g * g);
  idx.scatter.resize(static_cast<std::size_t>(g * g));
  for (std::int64_t y = 0; y < g; ++y)
    for (std::int64_t x = 0; x < g; ++x) {
      const std::int64_t row = ((y / win) * nw + x / win) * win * win + (y % win) * win + x % win;
      idx.gather[static_cast<std::size_t>(row)] = y * g + x;
      idx.scatter[static_cast<std::size_t>(y * g + x)] = row;
    }
  return idx;
}

Array upload_ids(const Context& ctx, const std::vector<std::int64_t>& ids) {
  Array a(ctx, DType::kI64, static_cast<std::int64_t>(ids.size()));
  a.upload(ids.data(), static_cast<std::int64_t>(ids.size()));
  return a;
}

}  // namespace

CudaVisionEncoder::CudaVisionEncoder(const Context& ctx, const WeightSource& w, const ModelDims& dims)
    : ctx_(ctx), dims_(dims) {
  const std::string sam = "model.sam_model";
  patch_ = matrix(ctx, w, sam + ".patch_embed.proj.weight");
  patch_b_ = vector(ctx, w, sam + ".patch_embed.proj.bias");
  sam_pos_host_ = w.get(sam + ".pos_embed");
  for (std::int64_t i = 0; i < dims.sam.layers; ++i) {
    const std::string p = std::format("{}.blocks.{}.", sam, i);
    SamBlock b;
    b.norm1_w = vector(ctx, w, p + "norm1.weight");
    b.norm1_b = vector(ctx, w, p + "norm1.bias");
    b.qkv = matrix(ctx, w, p + "attn.qkv.weight");
    b.qkv_b = vector(ctx, w, p + "attn.qkv.bias");
    b.proj = matrix(ctx, w, p + "attn.proj.weight");
    b.proj_b = vector(ctx, w, p + "attn.proj.bias");
    b.rel_h = w.get(p + "attn.rel_pos_h");
    b.rel_w = w.get(p + "attn.rel_pos_w");
    b.norm2_w = vector(ctx, w, p + "norm2.weight");
    b.norm2_b = vector(ctx, w, p + "norm2.bias");
    b.lin1 = matrix(ctx, w, p + "mlp.lin1.weight");
    b.lin1_b = vector(ctx, w, p + "mlp.lin1.bias");
    b.lin2 = matrix(ctx, w, p + "mlp.lin2.weight");
    b.lin2_b = vector(ctx, w, p + "mlp.lin2.bias");
    b.global = std::ranges::contains(dims.sam.global_layers, i);
    sam_blocks_.push_back(std::move(b));
  }
  neck0_ = matrix(ctx, w, sam + ".neck.0.weight");
  neck1_w_ = vector(ctx, w, sam + ".neck.1.weight");
  neck1_b_ = vector(ctx, w, sam + ".neck.1.bias");
  neck2_ = matrix(ctx, w, sam + ".neck.2.weight");
  neck3_w_ = vector(ctx, w, sam + ".neck.3.weight");
  neck3_b_ = vector(ctx, w, sam + ".neck.3.bias");
  net2_ = matrix(ctx, w, sam + ".net_2.weight");
  net3_ = matrix(ctx, w, sam + ".net_3.weight");

  const std::string clip = "model.vision_model";
  class_embedding_ = vector(ctx, w, clip + ".embeddings.class_embedding");
  clip_pos_host_ = w.get(clip + ".embeddings.position_embedding.weight");
  pre_ln_w_ = vector(ctx, w, clip + ".pre_layrnorm.weight");
  pre_ln_b_ = vector(ctx, w, clip + ".pre_layrnorm.bias");
  for (std::int64_t i = 0; i < dims.clip.layers; ++i) {
    const std::string p = std::format("{}.transformer.layers.{}.", clip, i);
    ClipLayer l;
    l.ln1_w = vector(ctx, w, p + "layer_norm1.weight");
    l.ln1_b = vector(ctx, w, p + "layer_norm1.bias");
    l.qkv = matrix(ctx, w, p + "self_attn.qkv_proj.weight");
    l.qkv_b = vector(ctx, w, p + "self_attn.qkv_proj.bias");
    l.out = matrix(ctx, w, p + "self_attn.out_proj.weight");
    l.out_b = vector(ctx, w, p + "self_attn.out_proj.bias");
    l.ln2_w = vector(ctx, w, p + "layer_norm2.weight");
    l.ln2_b = vector(ctx, w, p + "layer_norm2.bias");
    l.fc1 = matrix(ctx, w, p + "mlp.fc1.weight");
    l.fc1_b = vector(ctx, w, p + "mlp.fc1.bias");
    l.fc2 = matrix(ctx, w, p + "mlp.fc2.weight");
    l.fc2_b = vector(ctx, w, p + "mlp.fc2.bias");
    clip_layers_.push_back(std::move(l));
  }
  projector_ = matrix(ctx, w, "model.projector.layers.weight");
  projector_b_ = vector(ctx, w, "model.projector.layers.bias");
  newline_ = w.get("model.image_newline");
  separator_ = w.get("model.view_seperator");
}

const Array& CudaVisionEncoder::sam_positions(std::int64_t g) const {
  auto it = sam_pos_.find(g);
  if (it != sam_pos_.end()) return it->second;
  const std::int64_t src = sam_pos_host_.dim(1), c = sam_pos_host_.dim(3);
  Tensor pos({src, src, c}, std::vector<float>(sam_pos_host_.span().begin(), sam_pos_host_.span().end()));
  if (src != g) pos = reference::chw_to_hwc(reference::resize_bicubic_aa(reference::hwc_to_chw(pos), g, g));
  return sam_pos_.emplace(g, upload_tensor(ctx_, pos, DType::kF32)).first->second;
}

const Array& CudaVisionEncoder::clip_positions(std::int64_t g) const {
  auto it = clip_pos_.find(g);
  if (it != clip_pos_.end()) return it->second;
  const std::int64_t width = clip_pos_host_.dim(1);
  const auto src = static_cast<std::int64_t>(std::sqrt(static_cast<double>(clip_pos_host_.dim(0) - 1)));
  Tensor out = clip_pos_host_;
  if (src != g) {
    Tensor grid({src, src, width},
                std::vector<float>(clip_pos_host_.data() + width, clip_pos_host_.data() + clip_pos_host_.numel()));
    const Tensor resized = reference::chw_to_hwc(reference::resize_bicubic_aa(reference::hwc_to_chw(grid), g, g));
    out = Tensor({1 + g * g, width});
    std::copy_n(clip_pos_host_.data(), width, out.data());
    std::copy_n(resized.data(), resized.numel(), out.data() + width);
  }
  return clip_pos_.emplace(g, upload_tensor(ctx_, out, DType::kF32)).first->second;
}

const Array& CudaVisionEncoder::rel_table(const SamBlock& block, bool height, std::int64_t span) const {
  const auto key = std::make_tuple(&block, height, span);
  auto it = rel_tables_.find(key);
  if (it != rel_tables_.end()) return it->second;
  const Tensor table = relative_positions(height ? block.rel_h : block.rel_w, span);
  return rel_tables_.emplace(key, upload_tensor(ctx_, table, DType::kF32)).first->second;
}

void CudaVisionEncoder::sam_attention(const SamBlock& block, const Array& x, std::int64_t windows, std::int64_t h,
                                      std::int64_t w, Array& out) const {
  const std::int64_t c = dims_.sam.width, heads = dims_.sam.heads, hd = c / heads, n = h * w, rows = windows * n;
  Array qkv = f32(ctx_, rows * 3 * c);
  apply(ctx_, x, rows, block.qkv, &block.qkv_b, qkv);
  const Array& rh = rel_table(block, true, h);
  const Array& rw = rel_table(block, false, w);
  Array scores = f32(ctx_, windows * n * n), bias = f32(ctx_, windows * n * n), merged = f32(ctx_, rows * c);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  for (std::int64_t head = 0; head < heads; ++head) {
    const float* q = qkv.f32() + head * hd;
    gemm_nt(ctx_, n, n, hd, scale, q, 3 * c, n * 3 * c, q + c, 3 * c, n * 3 * c, scores.f32(), n, n * n, windows);
    for (std::int64_t win = 0; win < windows; ++win) {
      rel_pos_bias(ctx_, q + win * n * 3 * c, 3 * c, h, w, hd, rh.f32(), rw.f32(), bias.f32() + win * n * n);
    }
    add_inplace(ctx_, scores.f32(), bias.f32(), windows * n * n);
    softmax_rows(ctx_, scores.f32(), windows * n, n);
    gemm_nn(ctx_, n, hd, n, scores.f32(), n, n * n, q + 2 * c, 3 * c, n * 3 * c, merged.f32() + head * hd, c, n * c,
            windows);
  }
  apply(ctx_, merged, rows, block.proj, &block.proj_b, out);
}

void CudaVisionEncoder::sam_block(const SamBlock& block, Array& x, std::int64_t g) const {
  const std::int64_t c = dims_.sam.width, cells = g * g;
  Array normed = f32(ctx_, (cells + 1) * c);  // one extra zero row feeds window padding
  normed.zero();
  copy(ctx_, normed.f32(), x.f32(), cells * c);
  layer_norm(ctx_, normed.f32(), block.norm1_w.f32(), block.norm1_b.f32(), cells, c, kSamEps);
  Array attended = f32(ctx_, cells * c);
  if (block.global) {
    sam_attention(block, normed, 1, g, g, attended);
  } else {
    const std::int64_t win = dims_.sam.window;
    const WindowIndex idx = window_index(g, win);
    const Array gather = upload_ids(ctx_, idx.gather), scatter = upload_ids(ctx_, idx.scatter);
    const std::int64_t rows = idx.windows * win * win;
    Array windows = f32(ctx_, rows * c), out = f32(ctx_, rows * c);
    gather_rows(ctx_, normed.f32(), gather.i64(), rows, c, windows.f32());
    sam_attention(block, windows, idx.windows, win, win, out);
    gather_rows(ctx_, out.f32(), scatter.i64(), cells, c, attended.f32());
  }
  add_inplace(ctx_, x.f32(), attended.f32(), cells * c);

  Array hidden = f32(ctx_, cells * c), wide = f32(ctx_, cells * dims_.sam.mlp);
  copy(ctx_, hidden.f32(), x.f32(), cells * c);
  layer_norm(ctx_, hidden.f32(), block.norm2_w.f32(), block.norm2_b.f32(), cells, c, kSamEps);
  apply(ctx_, hidden, cells, block.lin1, &block.lin1_b, wide);
  gelu_erf(ctx_, wide.f32(), cells * dims_.sam.mlp);
  apply(ctx_, wide, cells, block.lin2, &block.lin2_b, hidden);
  add_inplace(ctx_, x.f32(), hidden.f32(), cells * c);
}

Array CudaVisionEncoder::sam_forward(const Array& image, std::int64_t size, std::int64_t& grid) const {
  const std::int64_t p = dims_.sam.patch, g = size / p, c = dims_.sam.width;
  Array cols = f32(ctx_, g * g * 3 * p * p);
  im2col(ctx_, image.f32(), 3, size, size, p, p, static_cast<int>(p), 0, cols.f32());
  Array x = f32(ctx_, g * g * c);
  apply(ctx_, cols, g * g, patch_, &patch_b_, x);
  add_inplace(ctx_, x.f32(), sam_positions(g).f32(), g * g * c);
  for (const auto& block : sam_blocks_) sam_block(block, x, g);

  // Neck and the two stride-2 convolutions, in HWC rows; convolutions with a
  // spatial kernel go through CHW for im2col.
  const std::int64_t nk = dims_.sam.neck;
  Array y = f32(ctx_, g * g * nk);
  apply(ctx_, x, g * g, neck0_, nullptr, y);
  layer_norm(ctx_, y.f32(), neck1_w_.f32(), neck1_b_.f32(), g * g, nk, kSamEps);
  auto conv3 = [&](const Array& hwc, std::int64_t side, std::int64_t in_c, const DeviceMatrix& w, int stride,
                   std::int64_t& out_side) {
    Array chw = f32(ctx_, side * side * in_c);
    hwc_to_chw(ctx_, hwc.f32(), side, side, in_c, chw.f32());
    out_side = (side + 2 - 3) / stride + 1;
    Array patches = f32(ctx_, out_side * out_side * in_c * 9);
    im2col(ctx_, chw.f32(), in_c, side, side, 3, 3, stride, 1, patches.f32());
    Array out = f32(ctx_, out_side * out_side * w.rows);
    apply(ctx_, patches, out_side * out_side, w, nullptr, out);
    return out;
  };
  std::int64_t s1 = 0, s2 = 0, s3 = 0;
  Array z = conv3(y, g, nk, neck2_, 1, s1);
  layer_norm(ctx_, z.f32(), neck3_w_.f32(), neck3_b_.f32(), s1 * s1, nk, kSamEps);
  Array n2 = conv3(z, s1, nk, net2_, 2, s2);
  Array n3 = conv3(n2, s2, dims_.sam.net2, net3_, 2, s3);
  grid = s3;
  return n3;
}

Array CudaVisionEncoder::clip_forward(const Array& cells, std::int64_t g) const {
  const std::int64_t c = dims_.clip.width, n = 1 + g * g, heads = dims_.clip.heads, hd = c / heads;
  Array x = f32(ctx_, n * c);
  copy(ctx_, x.f32(), class_embedding_.f32(), c);
  copy(ctx_, x.f32() + c, cells.f32(), g * g * c);
  add_inplace(ctx_, x.f32(), clip_positions(g).f32(), n * c);
  layer_norm(ctx_, x.f32(), pre_ln_w_.f32(), pre_ln_b_.f32(), n, c, kClipEps);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  Array normed = f32(ctx_, n * c), qkv = f32(ctx_, n * 3 * c), scores = f32(ctx_, heads * n * n);
  Array merged = f32(ctx_, n * c), proj = f32(ctx_, n * c), wide = f32(ctx_, n * dims_.clip.mlp);
  for (const auto& l : clip_layers_) {
    copy(ctx_, normed.f32(), x.f32(), n * c);
    layer_norm(ctx_, normed.f32(), l.ln1_w.f32(), l.ln1_b.f32(), n, c, kClipEps);
    apply(ctx_, normed, n, l.qkv, &l.qkv_b, qkv);
    gemm_nt(ctx_, n, n, hd, scale, qkv.f32(), 3 * c, hd, qkv.f32() + c, 3 * c, hd, scores.f32(), n, n * n, heads);
    softmax_rows(ctx_, scores.f32(), heads * n, n);
    gemm_nn(ctx_, n, hd, n, scores.f32(), n, n * n, qkv.f32() + 2 * c, 3 * c, hd, merged.f32(), c, hd, heads);
    apply(ctx_, merged, n, l.out, &l.out_b, proj);
    add_inplace(ctx_, x.f32(), proj.f32(), n * c);
    copy(ctx_, normed.f32(), x.f32(), n * c);
    layer_norm(ctx_, normed.f32(), l.ln2_w.f32(), l.ln2_b.f32(), n, c, kClipEps);
    apply(ctx_, normed, n, l.fc1, &l.fc1_b, wide);
    quick_gelu(ctx_, wide.f32(), n * dims_.clip.mlp);
    apply(ctx_, wide, n, l.fc2, &l.fc2_b, proj);
    add_inplace(ctx_, x.f32(), proj.f32(), n * c);
  }
  return x;
}

Tensor CudaVisionEncoder::encode_view(const Tensor& image) const {
  if (image.rank() != 3 || image.dim(0) != 3 || image.dim(1) % 64 != 0 || image.dim(1) != image.dim(2)) {
    throw std::invalid_argument("CudaVisionEncoder: image must be [3, S, S] with S a multiple of 64");
  }
  const Array pixels = upload_tensor(ctx_, image, DType::kF32);
  std::int64_t g = 0;
  const Array cells = sam_forward(pixels, image.dim(1), g);
  const Array clip = clip_forward(cells, g);
  const std::int64_t n = g * g, cw = dims_.clip.width, sw = dims_.sam.net3;
  Array joined = f32(ctx_, n * (cw + sw)), projected = f32(ctx_, n * projector_.rows);
  concat_cols(ctx_, clip.f32() + cw, cw, cells.f32(), sw, n, joined.f32());  // class token dropped
  apply(ctx_, joined, n, projector_, &projector_b_, projected);
  return Tensor({n, projector_.rows}, projected.download());
}

}  // namespace uocr::cuda
