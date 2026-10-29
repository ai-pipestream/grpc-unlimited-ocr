#include "model/block_router.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "backend/reference/parallel.h"

namespace uocr {
namespace {

constexpr float kActClamp = 1.0e4f;
constexpr float kEntmaxMasked = -1.0e4f;

float clamp_act(float v) { return std::clamp(v, -kActClamp, kActClamp); }

}  // namespace

Tensor block_summaries(const Tensor& idx_k, const Tensor& head_cls, std::int64_t block_size) {
  const std::int64_t s = idx_k.dim(0), d = idx_k.dim(1);
  const std::int64_t nblk = (s + block_size - 1) / block_size;
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  Tensor out({nblk, d});
  reference::parallel_for(nblk, [&](std::int64_t b) {
    const std::int64_t first = b * block_size, last = std::min(first + block_size, s);
    std::vector<float> w(static_cast<std::size_t>(last - first));
    float peak = -std::numeric_limits<float>::max();
    for (std::int64_t t = first; t < last; ++t) {
      float a = 0.0f;
      for (std::int64_t e = 0; e < d; ++e) a += idx_k.row(t)[e] * head_cls.data()[e];
      a = clamp_act(a * inv_sqrt_d);
      w[static_cast<std::size_t>(t - first)] = a;
      peak = std::max(peak, a);
    }
    float total = 0.0f;
    for (float& v : w) {
      v = std::exp(v - peak);
      total += v;
    }
    total = std::max(total, 1e-6f);
    float* o = out.row(b);
    for (std::int64_t t = first; t < last; ++t) {
      const float p = w[static_cast<std::size_t>(t - first)] / total;
      for (std::int64_t e = 0; e < d; ++e) o[e] += p * idx_k.row(t)[e];
    }
  });
  return out;
}

Tensor block_scores(const Tensor& idx_q, const Tensor& summaries, const std::vector<std::int64_t>& q_pos,
                    std::int64_t keys, const RouterParams& params) {
  const std::int64_t lq = idx_q.dim(0), d = idx_q.dim(1), nblk = summaries.dim(0);
  if (static_cast<std::int64_t>(q_pos.size()) != lq) {
    throw std::invalid_argument("block_scores: one position per query");
  }
  (void)keys;  // every summarised block holds at least one real key
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
  Tensor out({lq, nblk}, kScoreMasked);
  reference::parallel_for(lq, [&](std::int64_t i) {
    const std::int64_t last_remote = q_pos[static_cast<std::size_t>(i)] / params.block_size - params.local_blocks;
    for (std::int64_t b = 0; b <= std::min(last_remote, nblk - 1); ++b) {
      float z = 0.0f;
      for (std::int64_t e = 0; e < d; ++e) z += idx_q.row(i)[e] * summaries.row(b)[e];
      out.row(i)[b] = clamp_act(z * inv_sqrt_d);
    }
  });
  return out;
}

void entmax15(std::vector<float>& x) {
  if (x.empty()) return;
  const float peak = *std::ranges::max_element(x);
  for (float& v : x) v = (v - peak) / 2.0f;
  std::vector<float> sorted = x;
  std::ranges::sort(sorted, std::greater<>());
  float sum = 0.0f, sum_sq = 0.0f, tau_star = 0.0f;
  std::size_t support = 0;
  std::vector<float> tau(sorted.size());
  for (std::size_t k = 0; k < sorted.size(); ++k) {
    sum += sorted[k];
    sum_sq += sorted[k] * sorted[k];
    const auto rho = static_cast<float>(k + 1);
    const float mean = sum / rho, mean_sq = sum_sq / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = std::max((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - std::sqrt(delta);
    if (tau[k] <= sorted[k]) ++support;
  }
  tau_star = tau[std::max<std::size_t>(support, 1) - 1];
  for (float& v : x) {
    const float t = std::max(v - tau_star, 0.0f);
    v = t * t;
  }
}

BlockSelection select_blocks(const Tensor& scores, const std::vector<std::uint8_t>& candidate,
                             const RouterParams& params) {
  const std::int64_t lq = scores.dim(0), nblk = scores.dim(1);
  const std::int64_t m = std::max<std::int64_t>(1, std::min(params.top_m, nblk));
  const std::int64_t k = std::max<std::int64_t>(1, std::min(params.top_k, m));
  BlockSelection sel;
  sel.width = k;
  sel.block.assign(static_cast<std::size_t>(lq * k), 0);
  sel.bias.assign(static_cast<std::size_t>(lq * k), 0.0f);
  sel.valid.assign(static_cast<std::size_t>(lq * k), 0);
  reference::parallel_for(lq, [&](std::int64_t i) {
    const float* row = scores.row(i);
    std::vector<std::uint8_t> cand(static_cast<std::size_t>(nblk));
    bool any = false;
    for (std::int64_t b = 0; b < nblk; ++b) {
      const bool remote = row[b] > kScoreMasked * 0.5f;
      const bool allowed = candidate.empty() || candidate[static_cast<std::size_t>(i * nblk + b)] != 0;
      cand[static_cast<std::size_t>(b)] = remote && allowed;
      any = any || (remote && allowed);
    }
    if (!any) {
      for (std::int64_t b = 0; b < nblk; ++b) cand[static_cast<std::size_t>(b)] = row[b] > kScoreMasked * 0.5f;
    }
    std::vector<float> x(static_cast<std::size_t>(nblk));
    for (std::int64_t b = 0; b < nblk; ++b) {
      x[static_cast<std::size_t>(b)] = cand[static_cast<std::size_t>(b)] ? row[b] * params.idx_scale : kEntmaxMasked;
    }
    // Top m, descending; ties keep the lower block index first.
    std::vector<std::int64_t> order(static_cast<std::size_t>(nblk));
    std::iota(order.begin(), order.end(), 0);
    std::ranges::stable_sort(order, [&](std::int64_t a, std::int64_t b) {
      return x[static_cast<std::size_t>(a)] > x[static_cast<std::size_t>(b)];
    });
    std::vector<float> top(static_cast<std::size_t>(m));
    for (std::int64_t j = 0; j < m; ++j) top[static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])];
    entmax15(top);
    std::int64_t count = 0;
    float logp_sum = 0.0f;
    std::vector<float> logp(static_cast<std::size_t>(k));
    for (std::int64_t j = 0; j < k; ++j) {
      const std::int64_t b = order[static_cast<std::size_t>(j)];
      const std::size_t slot = static_cast<std::size_t>(i * k + j);
      sel.block[slot] = b;
      sel.valid[slot] = cand[static_cast<std::size_t>(b)];
      logp[static_cast<std::size_t>(j)] = std::log(std::max(top[static_cast<std::size_t>(j)], params.floor));
      if (sel.valid[slot] != 0) {
        logp_sum += logp[static_cast<std::size_t>(j)];
        ++count;
      }
    }
    const float mu = logp_sum / static_cast<float>(std::max<std::int64_t>(count, 1));
    for (std::int64_t j = 0; j < k; ++j) {
      const std::size_t slot = static_cast<std::size_t>(i * k + j);
      sel.bias[slot] = sel.valid[slot] != 0 ? (logp[static_cast<std::size_t>(j)] - mu) / params.sigma : 0.0f;
    }
  });
  return sel;
}

}  // namespace uocr
