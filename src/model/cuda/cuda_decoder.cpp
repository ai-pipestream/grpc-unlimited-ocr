#include "model/cuda/cuda_decoder.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

#include "backend/cuda/ops.h"
#include "backend/reference/ops.h"
#include "model/decoder.h"

namespace uocr::cuda {
namespace {

Array f32(const Context& ctx, std::int64_t n) { return Array(ctx, DType::kF32, n); }

void apply(const Context& ctx, const Array& x, std::int64_t rows, const DeviceMatrix& w, Array& y) {
  linear(ctx, x.f32(), rows, w.cols, w.data.bf16(), w.rows, nullptr, y.f32());
}

Array ids_on_device(const Context& ctx, const std::vector<std::int64_t>& ids) {
  Array a(ctx, DType::kI64, static_cast<std::int64_t>(ids.size()));
  a.upload(ids.data(), static_cast<std::int64_t>(ids.size()));
  return a;
}

}  // namespace

RouterParams CudaDecoder::default_router_params() { return Decoder::default_router(); }

CudaDecoder::CudaDecoder(const Context& ctx, const WeightSource& w, const DecoderDims& dims, RouterParams router)
    : ctx_(ctx), dims_(dims), router_(router), rope_(dims.head_dim) {
  embed_ = upload_tensor(ctx, w.get("model.embed_tokens.weight"), DType::kBF16);
  final_norm_ = vector(ctx, w, "model.norm.weight");
  lm_head_ = matrix(ctx, w, "lm_head.weight");
  inv_tok_ = f32(ctx, static_cast<std::int64_t>(rope_.inv_tok().size()));
  inv_tok_.upload(rope_.inv_tok().data(), inv_tok_.count());
  inv_page_ = f32(ctx, static_cast<std::int64_t>(rope_.inv_page().size()));
  inv_page_.upload(rope_.inv_page().data(), inv_page_.count());
  auto expert = [&](const std::string& p) {
    return Expert{matrix(ctx, w, p + ".gate_proj.weight"), matrix(ctx, w, p + ".up_proj.weight"),
                  matrix(ctx, w, p + ".down_proj.weight")};
  };
  for (std::int64_t i = 0; i < dims.layers; ++i) {
    const std::string p = std::format("model.layers.{}.", i);
    const std::string r = p + "self_attn.dash_router.";
    Layer l;
    l.input_norm = vector(ctx, w, p + "input_layernorm.weight");
    l.post_norm = vector(ctx, w, p + "post_attention_layernorm.weight");
    l.q = matrix(ctx, w, p + "self_attn.q_proj.weight");
    l.k = matrix(ctx, w, p + "self_attn.k_proj.weight");
    l.v = matrix(ctx, w, p + "self_attn.v_proj.weight");
    l.o = matrix(ctx, w, p + "self_attn.o_proj.weight");
    l.router_q = matrix(ctx, w, r + "q_proj.weight");
    l.router_k = matrix(ctx, w, r + "k_proj.weight");
    l.router_q_norm = vector(ctx, w, r + "q_norm.weight");
    l.router_k_norm = vector(ctx, w, r + "k_norm.weight");
    l.head_cls = vector(ctx, w, r + "head_cls");
    if (i < dims.first_moe_layer) {
      l.dense = expert(p + "mlp");
    } else {
      // The gate scores in fp32, as upstream does, so expert choice is not
      // perturbed by bf16 inputs.
      l.moe_gate_f32 = vector(ctx, w, p + "mlp.gate.weight");
      for (std::int64_t e = 0; e < dims.routed_experts; ++e) l.experts.push_back(expert(std::format("{}mlp.experts.{}", p, e)));
      l.shared = expert(p + "mlp.shared_experts");
    }
    layers_.push_back(std::move(l));
  }
}

Tensor CudaDecoder::embed(const std::vector<std::int64_t>& ids) const {
  const auto n = static_cast<std::int64_t>(ids.size());
  const Array idx = ids_on_device(ctx_, ids);
  Array out = f32(ctx_, n * dims_.hidden);
  gather_rows_bf16(ctx_, embed_.bf16(), idx.i64(), n, dims_.hidden, out.f32());
  return Tensor({n, dims_.hidden}, out.download());
}

void CudaDecoder::reserve(std::int64_t keys) {
  if (keys <= capacity_) return;
  const std::int64_t width = dims_.heads * dims_.head_dim;
  const std::int64_t capacity = std::max<std::int64_t>(keys, capacity_ * 2 + 1024);
  cache_.resize(layers_.size());
  for (auto& c : cache_) {
    auto grow = [&](Array& a, std::int64_t cols) {
      Array bigger = f32(ctx_, capacity * cols);
      if (a.count() > 0) copy(ctx_, bigger.f32(), a.f32(), length_ * cols);
      a = std::move(bigger);
    };
    grow(c.k, width);
    grow(c.v, width);
    grow(c.router_k, dims_.index_dim);
  }
  capacity_ = capacity;
}

void CudaDecoder::mlp(const Expert& e, const Array& x, std::int64_t rows, Array& out) const {
  Array g = f32(ctx_, rows * e.gate.rows), u = f32(ctx_, rows * e.up.rows);
  apply(ctx_, x, rows, e.gate, g);
  apply(ctx_, x, rows, e.up, u);
  silu_mul(ctx_, g.f32(), u.f32(), rows * e.gate.rows);
  apply(ctx_, g, rows, e.down, out);
}

void CudaDecoder::moe(const Layer& layer, const Array& h, std::int64_t rows, Array& out) const {
  const std::int64_t hidden = dims_.hidden, n_exp = dims_.routed_experts;
  Array logits = f32(ctx_, rows * n_exp);
  gemm_nt(ctx_, rows, n_exp, hidden, 1.0f, h.f32(), hidden, 0, layer.moe_gate_f32.f32(), hidden, 0, logits.f32(), n_exp, 0,
          1);
  Tensor scores({rows, n_exp}, logits.download());
  reference::softmax_rows(scores);
  // Same choice rule as the reference backend: top k by score, lower index on a tie.
  std::vector<std::vector<std::int64_t>> tokens_of(static_cast<std::size_t>(n_exp));
  std::vector<std::vector<float>> weight_of(static_cast<std::size_t>(n_exp));
  for (std::int64_t t = 0; t < rows; ++t) {
    std::vector<std::int64_t> order(static_cast<std::size_t>(n_exp));
    std::iota(order.begin(), order.end(), 0);
    const float* s = scores.row(t);
    std::ranges::partial_sort(order, order.begin() + dims_.experts_per_token,
                              [&](std::int64_t a, std::int64_t b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    for (std::int64_t j = 0; j < dims_.experts_per_token; ++j) {
      const std::int64_t e = order[static_cast<std::size_t>(j)];
      tokens_of[static_cast<std::size_t>(e)].push_back(t);
      weight_of[static_cast<std::size_t>(e)].push_back(s[e]);
    }
  }
  mlp(*layer.shared, h, rows, out);
  Array routed = f32(ctx_, rows * hidden);
  routed.zero();
  for (std::int64_t e = 0; e < n_exp; ++e) {
    const auto& toks = tokens_of[static_cast<std::size_t>(e)];
    if (toks.empty()) continue;
    const auto n = static_cast<std::int64_t>(toks.size());
    const Array idx = ids_on_device(ctx_, toks);
    Array wts = f32(ctx_, n);
    wts.upload(weight_of[static_cast<std::size_t>(e)].data(), n);
    Array in = f32(ctx_, n * hidden), res = f32(ctx_, n * hidden);
    gather_rows(ctx_, h.f32(), idx.i64(), n, hidden, in.f32());
    mlp(layer.experts[static_cast<std::size_t>(e)], in, n, res);
    scatter_add_rows(ctx_, routed.f32(), idx.i64(), wts.f32(), res.f32(), n, hidden);
  }
  add_inplace(ctx_, out.f32(), routed.f32(), rows * hidden);
}

void CudaDecoder::attention(const Layer& layer, std::size_t index, const Array& h, std::int64_t rows,
                            std::int64_t first_pos, const Array& tok, const Array& page,
                            const std::vector<std::vector<std::uint8_t>>& candidates, Array& out) {
  const std::int64_t width = dims_.heads * dims_.head_dim, hd = dims_.head_dim, d = dims_.index_dim;
  Cache& cache = cache_[index];
  float* k_rows = cache.k.f32() + first_pos * width;
  float* v_rows = cache.v.f32() + first_pos * width;
  float* rk_rows = cache.router_k.f32() + first_pos * d;
  Array q = f32(ctx_, rows * width), rq = f32(ctx_, rows * d);
  linear(ctx_, h.f32(), rows, layer.q.cols, layer.q.data.bf16(), width, nullptr, q.f32());
  linear(ctx_, h.f32(), rows, layer.k.cols, layer.k.data.bf16(), width, nullptr, k_rows);
  linear(ctx_, h.f32(), rows, layer.v.cols, layer.v.data.bf16(), width, nullptr, v_rows);
  linear(ctx_, h.f32(), rows, layer.router_q.cols, layer.router_q.data.bf16(), d, nullptr, rq.f32());
  linear(ctx_, h.f32(), rows, layer.router_k.cols, layer.router_k.data.bf16(), d, nullptr, rk_rows);
  router_norm(ctx_, rq.f32(), layer.router_q_norm.f32(), rows, d, static_cast<float>(dims_.rms_norm_eps));
  router_norm(ctx_, rk_rows, layer.router_k_norm.f32(), rows, d, static_cast<float>(dims_.rms_norm_eps));
  const auto n_tok = static_cast<std::int64_t>(rope_.inv_tok().size());
  for (float* x : {q.f32(), k_rows}) {
    rope(ctx_, x, rows, dims_.heads, hd, tok.i64(), page.i64(), inv_tok_.f32(), n_tok, inv_page_.f32(), 1,
         rope_.page_mod());
  }
  for (float* x : {rq.f32(), rk_rows}) {
    rope(ctx_, x, rows, 1, d, tok.i64(), page.i64(), inv_tok_.f32(), n_tok, inv_page_.f32(), 1, rope_.page_mod());
  }

  // Route: block summaries and scores on the device, the selection on the host.
  const std::int64_t keys = first_pos + rows;
  const std::int64_t nblk = (keys + router_.block_size - 1) / router_.block_size;
  Array summaries = f32(ctx_, nblk * d), z = f32(ctx_, rows * nblk);
  block_summaries(ctx_, cache.router_k.f32(), keys, d, layer.head_cls.f32(), router_.block_size, summaries.f32());
  gemm_nt(ctx_, rows, nblk, d, 1.0f / std::sqrt(static_cast<float>(d)), rq.f32(), d, 0, summaries.f32(), d, 0, z.f32(),
          nblk, 0, 1);
  Tensor scores({rows, nblk}, z.download());
  mask_block_scores(scores, first_pos, router_);
  const BlockSelection sel = select_blocks(scores, flatten_candidates(candidates, nblk), router_);
  const std::vector<float> mask_host = key_mask(sel, rows, first_pos, keys, router_);
  Array mask = f32(ctx_, rows * keys);
  mask.upload(mask_host.data(), rows * keys);

  // Attend: per-head scores over every cached key, masked to the plan.
  Array s = f32(ctx_, dims_.heads * rows * keys), merged = f32(ctx_, rows * width);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  gemm_nt(ctx_, rows, keys, hd, scale, q.f32(), width, hd, cache.k.f32(), width, hd, s.f32(), keys, rows * keys,
          dims_.heads);
  scale_add_mask(ctx_, s.f32(), 1.0f, mask.f32(), dims_.heads, rows, keys);
  softmax_rows(ctx_, s.f32(), dims_.heads * rows, keys);
  gemm_nn(ctx_, rows, hd, keys, s.f32(), keys, rows * keys, cache.v.f32(), width, hd, merged.f32(), width, hd,
          dims_.heads);
  apply(ctx_, merged, rows, layer.o, out);
}

std::vector<float> CudaDecoder::run(Array& x, std::int64_t rows, std::int64_t first_pos,
                                    const std::vector<RopeCoord>& coords,
                                    const std::vector<std::vector<std::uint8_t>>& candidates) {
  reserve(first_pos + rows);
  std::vector<std::int64_t> tok, page;
  for (const auto& c : coords) {
    tok.push_back(c.tok);
    page.push_back(c.page);
  }
  const Array tok_d = ids_on_device(ctx_, tok), page_d = ids_on_device(ctx_, page);
  const std::int64_t hidden = dims_.hidden;
  const float eps = static_cast<float>(dims_.rms_norm_eps);
  Array h = f32(ctx_, rows * hidden), delta = f32(ctx_, rows * hidden);
  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    copy(ctx_, h.f32(), x.f32(), rows * hidden);
    rms_norm(ctx_, h.f32(), layer.input_norm.f32(), rows, hidden, eps);
    attention(layer, li, h, rows, first_pos, tok_d, page_d, candidates, delta);
    add_inplace(ctx_, x.f32(), delta.f32(), rows * hidden);
    copy(ctx_, h.f32(), x.f32(), rows * hidden);
    rms_norm(ctx_, h.f32(), layer.post_norm.f32(), rows, hidden, eps);
    if (layer.dense) {
      mlp(*layer.dense, h, rows, delta);
    } else {
      moe(layer, h, rows, delta);
    }
    add_inplace(ctx_, x.f32(), delta.f32(), rows * hidden);
  }
  length_ = first_pos + rows;
  Array last = f32(ctx_, hidden), logits = f32(ctx_, lm_head_.rows);
  copy(ctx_, last.f32(), x.f32() + (rows - 1) * hidden, hidden);
  rms_norm(ctx_, last.f32(), final_norm_.f32(), 1, hidden, eps);
  apply(ctx_, last, 1, lm_head_, logits);
  return logits.download();
}

std::vector<float> CudaDecoder::prefill(const Tensor& embeds, const PromptLayout& layout) {
  length_ = 0;
  const std::int64_t s = embeds.dim(0);
  const BlockPages blocks = block_image_pages(layout, router_.block_size);
  const std::int64_t nblk = (s + router_.block_size - 1) / router_.block_size;
  std::vector<std::vector<std::uint8_t>> candidates;
  for (std::int64_t i = 0; i < s; ++i) {
    candidates.push_back(page_candidates(blocks, layout.query_page[static_cast<std::size_t>(i)], nblk));
  }
  Array x = upload_tensor(ctx_, embeds, DType::kF32);
  return run(x, s, 0, layout.coords, candidates);
}

std::vector<float> CudaDecoder::step(std::int64_t token, RopeCoord coord, std::int64_t query_page,
                                     const BlockPages& prompt_blocks) {
  const std::int64_t pos = length_;
  const std::int64_t nblk = (pos + 1 + router_.block_size - 1) / router_.block_size;
  const Array idx = ids_on_device(ctx_, {token});
  Array x = f32(ctx_, dims_.hidden);
  gather_rows_bf16(ctx_, embed_.bf16(), idx.i64(), 1, dims_.hidden, x.f32());
  return run(x, 1, pos, {coord}, {page_candidates(prompt_blocks, query_page, nblk)});
}

}  // namespace uocr::cuda
