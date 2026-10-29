#include "model/feed_forward.h"

#include <algorithm>
#include <format>
#include <numeric>

#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"

namespace uocr {
namespace ref = reference;

GatedMlp::GatedMlp(const WeightSource& weights, const std::string& prefix)
    : gate_(weights.get(prefix + ".gate_proj.weight")),
      up_(weights.get(prefix + ".up_proj.weight")),
      down_(weights.get(prefix + ".down_proj.weight")) {}

Tensor GatedMlp::forward(const Tensor& x) const {
  Tensor g = ref::linear(x, gate_);
  const Tensor u = ref::linear(x, up_);
  ref::silu(g);
  for (std::int64_t i = 0; i < g.numel(); ++i) g.data()[i] *= u.data()[i];
  return ref::linear(g, down_);
}

MixtureOfExperts::MixtureOfExperts(const WeightSource& weights, const std::string& prefix, const DecoderDims& dims)
    : gate_(weights.get(prefix + ".gate.weight")),
      shared_(weights, prefix + ".shared_experts"),
      top_k_(dims.experts_per_token) {
  experts_.reserve(static_cast<std::size_t>(dims.routed_experts));
  for (std::int64_t e = 0; e < dims.routed_experts; ++e) {
    experts_.emplace_back(weights, std::format("{}.experts.{}", prefix, e));
  }
}

Tensor MixtureOfExperts::forward(const Tensor& x) const {
  const std::int64_t l = x.dim(0), hidden = x.dim(1), n_exp = gate_.dim(0);
  Tensor scores = ref::linear(x, gate_);
  ref::softmax_rows(scores);
  // Per token, its top_k experts and their (unnormalised) gate weights.
  std::vector<std::vector<std::int64_t>> tokens_of(static_cast<std::size_t>(n_exp));
  std::vector<std::vector<float>> weight_of(static_cast<std::size_t>(n_exp));
  for (std::int64_t t = 0; t < l; ++t) {
    std::vector<std::int64_t> order(static_cast<std::size_t>(n_exp));
    std::iota(order.begin(), order.end(), 0);
    std::ranges::partial_sort(order, order.begin() + top_k_, [&](std::int64_t a, std::int64_t b) {
      return scores.row(t)[a] > scores.row(t)[b] || (scores.row(t)[a] == scores.row(t)[b] && a < b);
    });
    for (std::int64_t j = 0; j < top_k_; ++j) {
      const std::int64_t e = order[static_cast<std::size_t>(j)];
      tokens_of[static_cast<std::size_t>(e)].push_back(t);
      weight_of[static_cast<std::size_t>(e)].push_back(scores.row(t)[e]);
    }
  }
  // Each expert runs once over the tokens routed to it; contributions are
  // added per token in expert order, so the sum order is fixed.
  std::vector<Tensor> outputs(static_cast<std::size_t>(n_exp));
  ref::parallel_for(n_exp, [&](std::int64_t e) {
    const auto& toks = tokens_of[static_cast<std::size_t>(e)];
    if (toks.empty()) return;
    Tensor in({static_cast<std::int64_t>(toks.size()), hidden});
    for (std::size_t i = 0; i < toks.size(); ++i) std::copy_n(x.row(toks[i]), hidden, in.row(static_cast<std::int64_t>(i)));
    outputs[static_cast<std::size_t>(e)] = experts_[static_cast<std::size_t>(e)].forward(in);
  });
  Tensor y = shared_.forward(x);
  Tensor routed({l, hidden});
  for (std::int64_t e = 0; e < n_exp; ++e) {
    const auto& toks = tokens_of[static_cast<std::size_t>(e)];
    for (std::size_t i = 0; i < toks.size(); ++i) {
      const float w = weight_of[static_cast<std::size_t>(e)][i];
      const float* src = outputs[static_cast<std::size_t>(e)].row(static_cast<std::int64_t>(i));
      float* dst = routed.row(toks[i]);
      for (std::int64_t c = 0; c < hidden; ++c) dst[c] += w * src[c];
    }
  }
  for (std::int64_t i = 0; i < y.numel(); ++i) y.data()[i] += routed.data()[i];
  return y;
}

FeedForward::FeedForward(const WeightSource& weights, const std::string& prefix, const DecoderDims& dims, bool dense) {
  if (dense) {
    dense_.emplace(weights, prefix);
  } else {
    moe_.emplace(weights, prefix, dims);
  }
}

Tensor FeedForward::forward(const Tensor& x) const { return dense_ ? dense_->forward(x) : moe_->forward(x); }

}  // namespace uocr
