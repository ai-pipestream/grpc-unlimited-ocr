#include "model/decoder.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"

namespace uocr {
namespace ref = reference;
namespace {

constexpr float kActClamp = 1.0e4f;

// The router's RMSNorm: scale by (1 + weight), finite and clamped.
void router_norm(Tensor& x, const Tensor& weight, float eps) {
  const std::int64_t c = x.shape().back();
  for (std::int64_t r = 0; r < x.rows(); ++r) {
    float* v = x.row(r);
    double ss = 0.0;
    for (std::int64_t i = 0; i < c; ++i) {
      if (!std::isfinite(v[i])) v[i] = 0.0f;
      ss += static_cast<double>(v[i]) * v[i];
    }
    const auto inv = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(c) + eps));
    for (std::int64_t i = 0; i < c; ++i) {
      const float n = std::clamp(v[i] * inv, -kActClamp, kActClamp);
      v[i] = std::clamp(n * (1.0f + weight.data()[i]), -kActClamp, kActClamp);
    }
  }
}

void append_rows(std::vector<float>& dst, const Tensor& src) { dst.insert(dst.end(), src.span().begin(), src.span().end()); }

void add_into(Tensor& x, const Tensor& y) {
  for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] += y.data()[i];
}

}  // namespace

std::vector<std::uint8_t> page_candidates(const BlockPages& blocks, std::int64_t page, std::int64_t nblk) {
  if (page < 0) return {};
  std::vector<std::uint8_t> out(static_cast<std::size_t>(nblk), 0);
  bool any = false;
  const auto known = std::min<std::int64_t>(nblk, static_cast<std::int64_t>(blocks.min_page.size()));
  for (std::int64_t b = 0; b < known; ++b) {
    const std::int64_t lo = blocks.min_page[static_cast<std::size_t>(b)], hi = blocks.max_page[static_cast<std::size_t>(b)];
    if (lo >= 0 && lo <= page && page <= hi) {
      out[static_cast<std::size_t>(b)] = 1;
      any = true;
    }
  }
  if (!any) return {};
  return out;
}

RouterParams Decoder::default_router() {
  RouterParams p;
  p.idx_scale = 128.0f / std::sqrt(128.0f);  // dash_scale_per_layer / sqrt(index_dim)
  return p;
}

Decoder::Decoder(const WeightSource& weights, const DecoderDims& dims, RouterParams router)
    : dims_(dims), router_(router), rope_(dims.head_dim) {
  embed_tokens_ = weights.get("model.embed_tokens.weight");
  final_norm_ = weights.get("model.norm.weight");
  lm_head_ = weights.get("lm_head.weight");
  for (std::int64_t i = 0; i < dims.layers; ++i) {
    const std::string p = std::format("model.layers.{}.", i);
    const std::string r = p + "self_attn.dash_router.";
    layers_.push_back(Layer{
        weights.get(p + "input_layernorm.weight"), weights.get(p + "post_attention_layernorm.weight"),
        weights.get(p + "self_attn.q_proj.weight"), weights.get(p + "self_attn.k_proj.weight"),
        weights.get(p + "self_attn.v_proj.weight"), weights.get(p + "self_attn.o_proj.weight"),
        Router{weights.get(r + "q_proj.weight"), weights.get(r + "k_proj.weight"), weights.get(r + "q_norm.weight"),
               weights.get(r + "k_norm.weight"), weights.get(r + "head_cls")},
        FeedForward(weights, p + "mlp", dims, i < dims.first_moe_layer)});
  }
}

Tensor Decoder::embed(const std::vector<std::int64_t>& ids) const {
  const std::int64_t hidden = embed_tokens_.dim(1);
  Tensor out({static_cast<std::int64_t>(ids.size()), hidden});
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::copy_n(embed_tokens_.row(ids[i]), hidden, out.row(static_cast<std::int64_t>(i)));
  }
  return out;
}

Tensor Decoder::attention(const Layer& layer, const Tensor& x, const std::vector<RopeCoord>& coords,
                          const std::vector<std::vector<std::uint8_t>>& candidates, DecoderCache::Layer& cache,
                          std::int64_t first_pos) const {
  const std::int64_t l = x.dim(0), width = dims_.heads * dims_.head_dim, hd = dims_.head_dim;
  Tensor q = ref::linear(x, layer.q), k = ref::linear(x, layer.k);
  const Tensor v = ref::linear(x, layer.v);
  rope_.apply(q, coords);
  rope_.apply(k, coords);
  Tensor rq = ref::linear(x, layer.router.q), rk = ref::linear(x, layer.router.k);
  router_norm(rq, layer.router.q_norm, static_cast<float>(dims_.rms_norm_eps));
  router_norm(rk, layer.router.k_norm, static_cast<float>(dims_.rms_norm_eps));
  rope_.apply(rq, coords);
  rope_.apply(rk, coords);
  append_rows(cache.k, k);
  append_rows(cache.v, v);
  append_rows(cache.router_k, rk);

  const std::int64_t keys = static_cast<std::int64_t>(cache.k.size()) / width;
  const Tensor all_rk({keys, dims_.index_dim}, cache.router_k);
  const Tensor summaries = block_summaries(all_rk, layer.router.head_cls, router_.block_size);
  const std::int64_t nblk = summaries.dim(0);
  std::vector<std::int64_t> q_pos(static_cast<std::size_t>(l));
  for (std::int64_t i = 0; i < l; ++i) q_pos[static_cast<std::size_t>(i)] = first_pos + i;
  const Tensor scores = block_scores(rq, summaries, q_pos, keys, router_);
  std::vector<std::uint8_t> cand;
  bool any_cand = false;
  for (const auto& c : candidates) any_cand = any_cand || !c.empty();
  if (any_cand) {
    cand.assign(static_cast<std::size_t>(l * nblk), 1);
    for (std::int64_t i = 0; i < l; ++i) {
      const auto& c = candidates[static_cast<std::size_t>(i)];
      if (c.empty()) continue;
      for (std::int64_t b = 0; b < nblk; ++b) {
        cand[static_cast<std::size_t>(i * nblk + b)] = b < static_cast<std::int64_t>(c.size()) ? c[static_cast<std::size_t>(b)] : 0;
      }
    }
  }
  const BlockSelection sel = select_blocks(scores, cand, router_);

  // Keys attended by query i: its local window without bias, plus its
  // selected blocks with their bias; everything else masked.
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::int64_t window = router_.local_window();
  Tensor out({l, width});
  ref::parallel_for(l, [&](std::int64_t i) {
    const std::int64_t pos = first_pos + i;
    std::vector<float> block_bias(static_cast<std::size_t>(nblk), -std::numeric_limits<float>::infinity());
    for (std::int64_t j = 0; j < sel.width; ++j) {
      const std::size_t slot = static_cast<std::size_t>(i * sel.width + j);
      if (sel.valid[slot] != 0) block_bias[static_cast<std::size_t>(sel.block[slot])] = sel.bias[slot];
    }
    std::vector<float> mask(static_cast<std::size_t>(pos + 1));
    for (std::int64_t j = 0; j <= pos; ++j) {
      mask[static_cast<std::size_t>(j)] =
          j > pos - window ? 0.0f : block_bias[static_cast<std::size_t>(j / router_.block_size)];
    }
    std::vector<float> s(static_cast<std::size_t>(pos + 1));
    for (std::int64_t h = 0; h < dims_.heads; ++h) {
      const float* qh = q.row(i) + h * hd;
      float peak = -std::numeric_limits<float>::infinity();
      for (std::int64_t j = 0; j <= pos; ++j) {
        if (mask[static_cast<std::size_t>(j)] == -std::numeric_limits<float>::infinity()) {
          s[static_cast<std::size_t>(j)] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float* kj = cache.k.data() + j * width + h * hd;
        float dot = 0.0f;
        for (std::int64_t e = 0; e < hd; ++e) dot += qh[e] * kj[e];
        s[static_cast<std::size_t>(j)] = dot * scale + mask[static_cast<std::size_t>(j)];
        peak = std::max(peak, s[static_cast<std::size_t>(j)]);
      }
      double total = 0.0;
      for (float& e : s) {
        e = e == -std::numeric_limits<float>::infinity() ? 0.0f : std::exp(e - peak);
        total += e;
      }
      float* o = out.row(i) + h * hd;
      const auto inv = static_cast<float>(1.0 / total);
      for (std::int64_t j = 0; j <= pos; ++j) {
        const float p = s[static_cast<std::size_t>(j)] * inv;
        if (p == 0.0f) continue;
        const float* vj = cache.v.data() + j * width + h * hd;
        for (std::int64_t e = 0; e < hd; ++e) o[e] += p * vj[e];
      }
    }
  });
  return ref::linear(out, layer.o);
}

std::vector<float> Decoder::prefill(const Tensor& embeds, const PromptLayout& layout, DecoderCache& cache) const {
  cache.layers.assign(layers_.size(), {});
  cache.length = 0;
  const std::int64_t s = embeds.dim(0);
  const BlockPages blocks = block_image_pages(layout, router_.block_size);
  const std::int64_t nblk = (s + router_.block_size - 1) / router_.block_size;
  std::vector<std::vector<std::uint8_t>> candidates;
  candidates.reserve(static_cast<std::size_t>(s));
  for (std::int64_t i = 0; i < s; ++i) {
    candidates.push_back(page_candidates(blocks, layout.query_page[static_cast<std::size_t>(i)], nblk));
  }
  Tensor x = embeds;
  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    Tensor h = x;
    ref::rms_norm(h, layer.input_norm, static_cast<float>(dims_.rms_norm_eps));
    add_into(x, attention(layer, h, layout.coords, candidates, cache.layers[li], 0));
    h = x;
    ref::rms_norm(h, layer.post_norm, static_cast<float>(dims_.rms_norm_eps));
    add_into(x, layer.ffn.forward(h));
  }
  cache.length = s;
  const Tensor last({1, x.dim(1)}, std::vector<float>(x.row(s - 1), x.row(s - 1) + x.dim(1)));
  return logits(last);
}

std::vector<float> Decoder::step(std::int64_t token, RopeCoord coord, std::int64_t query_page,
                                 const BlockPages& prompt_blocks, DecoderCache& cache) const {
  const std::int64_t pos = cache.length;
  const std::int64_t nblk = (pos + 1 + router_.block_size - 1) / router_.block_size;
  const std::vector<std::vector<std::uint8_t>> candidates{page_candidates(prompt_blocks, query_page, nblk)};
  Tensor x = embed({token});
  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    Tensor h = x;
    ref::rms_norm(h, layer.input_norm, static_cast<float>(dims_.rms_norm_eps));
    add_into(x, attention(layer, h, {coord}, candidates, cache.layers[li], pos));
    h = x;
    ref::rms_norm(h, layer.post_norm, static_cast<float>(dims_.rms_norm_eps));
    add_into(x, layer.ffn.forward(h));
  }
  cache.length = pos + 1;
  return logits(x);
}

std::vector<float> Decoder::logits(const Tensor& hidden_last) const {
  Tensor h = hidden_last;
  ref::rms_norm(h, final_norm_, static_cast<float>(dims_.rms_norm_eps));
  const Tensor out = ref::linear(h, lm_head_);
  return {out.span().begin(), out.span().end()};
}

}  // namespace uocr
