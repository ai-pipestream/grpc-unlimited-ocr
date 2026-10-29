#include "model/cuda/cuda_decoder.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

#include "backend/cuda/decode_ops.h"
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

// Concatenates one projection of every expert into a single bf16 array.
Array stack(const Context& ctx, const WeightSource& w, const std::string& prefix, const std::string& proj,
            std::int64_t experts, std::int64_t per_expert) {
  Array out(ctx, DType::kBF16, experts * per_expert);
  for (std::int64_t e = 0; e < experts; ++e) {
    const Tensor t = w.get(std::format("{}.{}.{}.weight", prefix, e, proj));
    out.upload(t.data(), per_expert, e * per_expert);
  }
  return out;
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
      l.experts = stack_experts(ctx, w, p + "mlp.experts", dims);
      l.shared = expert(p + "mlp.shared_experts");
    }
    layers_.push_back(std::move(l));
  }
}

CudaDecoder::ExpertStack CudaDecoder::stack_experts(const Context& ctx, const WeightSource& w, const std::string& prefix,
                                                   const DecoderDims& dims) {
  const std::int64_t per = dims.expert_ffn * dims.hidden;
  return ExpertStack{stack(ctx, w, prefix, "gate_proj", dims.routed_experts, per),
                     stack(ctx, w, prefix, "up_proj", dims.routed_experts, per),
                     stack(ctx, w, prefix, "down_proj", dims.routed_experts, per), dims.expert_ffn};
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
    const std::int64_t blocks = (capacity + router_.block_size - 1) / router_.block_size;
    Array summaries = f32(ctx_, blocks * dims_.index_dim);
    const std::int64_t used = (length_ + router_.block_size - 1) / router_.block_size;
    if (c.summaries.count() > 0 && used > 0) copy(ctx_, summaries.f32(), c.summaries.f32(), used * dims_.index_dim);
    c.summaries = std::move(summaries);
  }
  capacity_ = capacity;
  capacity_blocks_ = (capacity + router_.block_size - 1) / router_.block_size;
  graph_.reset();  // buffers moved: the captured step must be recorded again
}

void CudaDecoder::mlp(const Expert& e, const Array& x, std::int64_t rows, Array& out) const {
  Array g = f32(ctx_, rows * e.gate.rows), u = f32(ctx_, rows * e.up.rows);
  apply(ctx_, x, rows, e.gate, g);
  apply(ctx_, x, rows, e.up, u);
  silu_mul(ctx_, g.f32(), u.f32(), rows * e.gate.rows);
  apply(ctx_, g, rows, e.down, out);
}

void CudaDecoder::mlp(MlpView e, const Array& x, std::int64_t rows, Array& out) const {
  const std::int64_t hidden = dims_.hidden;
  Array g = f32(ctx_, rows * e.ffn), u = f32(ctx_, rows * e.ffn);
  linear(ctx_, x.f32(), rows, hidden, e.gate, e.ffn, nullptr, g.f32());
  linear(ctx_, x.f32(), rows, hidden, e.up, e.ffn, nullptr, u.f32());
  silu_mul(ctx_, g.f32(), u.f32(), rows * e.ffn);
  linear(ctx_, g.f32(), rows, e.ffn, e.down, hidden, nullptr, out.f32());
}

void CudaDecoder::moe(const Layer& layer, const Array& h, std::int64_t rows, Array& out) const {
  const std::int64_t hidden = dims_.hidden, n_exp = dims_.routed_experts;
  Array logits = f32(ctx_, rows * n_exp);
  gemm_nt(ctx_, rows, n_exp, hidden, 1.0f, h.f32(), hidden, 0, layer.moe_gate_f32.f32(), hidden, 0, logits.f32(), n_exp, 0,
          1);
  const ExpertStack& stack = layer.experts;
  if (rows == 1) {
    // Decode: routing and the six experts stay on the device.
    const std::int64_t k = dims_.experts_per_token;
    Array ids(ctx_, DType::kI64, k), wts = f32(ctx_, k), routed = f32(ctx_, hidden);
    expert_topk(ctx_, logits.f32(), 1, n_exp, k, ids.i64(), wts.f32());
    expert_ffn_one(ctx_, h.f32(), hidden, stack.ffn, stack.gate.bf16(), stack.up.bf16(), stack.down.bf16(), ids.i64(),
                   wts.f32(), k, routed.f32());
    mlp(*layer.shared, h, rows, out);
    add_inplace(ctx_, out.f32(), routed.f32(), hidden);
    return;
  }
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
    const std::int64_t per = stack.ffn * hidden;
    mlp(MlpView{stack.gate.bf16() + e * per, stack.up.bf16() + e * per, stack.down.bf16() + e * per, stack.ffn}, in, n,
        res);
    scatter_add_rows(ctx_, routed.f32(), idx.i64(), wts.f32(), res.f32(), n, hidden);
  }
  add_inplace(ctx_, out.f32(), routed.f32(), rows * hidden);
}

void CudaDecoder::attention(const Layer& layer, std::size_t index, const Array& h, std::int64_t rows,
                            std::int64_t first_pos, const Array& tok, const Array& page, QueryPages pages,
                            Array& out) {
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
  const std::int64_t* tok_rows = tok.count() > 0 ? tok.i64() : nullptr;
  const std::int64_t* page_rows = page.count() > 0 ? page.i64() : nullptr;
  for (float* x : {q.f32(), k_rows}) {
    rope(ctx_, x, rows, dims_.heads, hd, tok_rows, page_rows, inv_tok_.f32(), n_tok, inv_page_.f32(), 1,
         rope_.page_mod(), one_coord_.tok, one_coord_.page);
  }
  for (float* x : {rq.f32(), rk_rows}) {
    rope(ctx_, x, rows, 1, d, tok_rows, page_rows, inv_tok_.f32(), n_tok, inv_page_.f32(), 1, rope_.page_mod(),
         one_coord_.tok, one_coord_.page);
  }

  // Route on the device: block summaries, scores, then the selection and
  // the key mask, the same rule as the host plan.
  const std::int64_t keys = first_pos + rows;
  const std::int64_t nblk = (keys + router_.block_size - 1) / router_.block_size;
  Array z = f32(ctx_, rows * nblk), mask = f32(ctx_, rows * keys);
  block_summaries(ctx_, cache.router_k.f32(), keys, d, layer.head_cls.f32(), router_.block_size, cache.summaries.f32());
  gemm_nt(ctx_, rows, nblk, d, 1.0f / std::sqrt(static_cast<float>(d)), rq.f32(), d, 0, cache.summaries.f32(), d, 0,
          z.f32(), nblk, 0, 1);
  route_blocks(ctx_, z.f32(), rows, nblk, first_pos, keys, block_min_.i64(), block_max_.i64(), prompt_blocks_, pages.rows,
               pages.one,
               {router_.block_size, router_.local_blocks, router_.top_m, router_.top_k, router_.idx_scale, router_.floor,
                router_.sigma},
               mask.f32());

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
                                    const std::vector<RopeCoord>& coords, QueryPages pages) {
  reserve(first_pos + rows);
  // A single row passes its coordinate by value: no upload, no sync.
  std::vector<std::int64_t> tok, page;
  if (rows > 1) {
    for (const auto& c : coords) {
      tok.push_back(c.tok);
      page.push_back(c.page);
    }
  }
  const Array tok_d = rows > 1 ? ids_on_device(ctx_, tok) : Array();
  const Array page_d = rows > 1 ? ids_on_device(ctx_, page) : Array();
  one_coord_ = coords.front();
  const std::int64_t hidden = dims_.hidden;
  const float eps = static_cast<float>(dims_.rms_norm_eps);
  Array h = f32(ctx_, rows * hidden), delta = f32(ctx_, rows * hidden);
  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    copy(ctx_, h.f32(), x.f32(), rows * hidden);
    rms_norm(ctx_, h.f32(), layer.input_norm.f32(), rows, hidden, eps);
    attention(layer, li, h, rows, first_pos, tok_d, page_d, pages, delta);
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
  prompt_blocks_ = static_cast<std::int64_t>(blocks.min_page.size());
  block_min_ = ids_on_device(ctx_, blocks.min_page);
  block_max_ = ids_on_device(ctx_, blocks.max_page);
  const Array query_pages = ids_on_device(ctx_, layout.query_page);
  Array x = upload_tensor(ctx_, embeds, DType::kF32);
  return run(x, s, 0, layout.coords, QueryPages{query_pages.i64(), -1});
}

std::vector<float> CudaDecoder::step(std::int64_t token, RopeCoord coord, std::int64_t query_page,
                                     const BlockPages& prompt_blocks) {
  (void)prompt_blocks;  // the prompt's block pages are on the device since prefill
  const std::int64_t pos = length_;
  reserve(pos + 1);
  if (!state_host_) {
    state_host_ = std::make_unique<PinnedBuffer>(sizeof(StepState));
    logits_host_ = std::make_unique<PinnedBuffer>(static_cast<std::size_t>(lm_head_.rows) * 4);
  }
  if (!graph_) {
    const std::int64_t width = dims_.heads * dims_.head_dim, hidden = dims_.hidden, d = dims_.index_dim;
    step_ = StepBuffers{Array(ctx_, DType::kI64, 5), f32(ctx_, hidden), f32(ctx_, hidden), f32(ctx_, hidden),
                        f32(ctx_, width), f32(ctx_, width), f32(ctx_, width), f32(ctx_, d), f32(ctx_, d),
                        f32(ctx_, capacity_blocks_), f32(ctx_, capacity_), f32(ctx_, width),
                        f32(ctx_, attend_scratch_floats(dims_.heads, dims_.head_dim, capacity_)),
                        f32(ctx_, lm_head_.rows)};
    ctx_.synchronize();
    ctx_.begin_capture();
    decode_body();
    graph_ = ctx_.end_capture();
  }
  *static_cast<StepState*>(state_host_->data()) = StepState{token, pos, coord.tok, coord.page, query_page};
  copy_to_device(ctx_, step_.state.data(), *state_host_, sizeof(StepState));
  graph_->launch();
  copy_to_host(ctx_, *logits_host_, step_.logits.data(), logits_host_->bytes());
  ctx_.synchronize();
  length_ = pos + 1;
  const auto* l = static_cast<const float*>(logits_host_->data());
  return {l, l + lm_head_.rows};
}

}  // namespace uocr::cuda
