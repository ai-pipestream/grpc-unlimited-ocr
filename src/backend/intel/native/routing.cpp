#include <algorithm>
#include <cmath>

#include "backend/intel/native/context_impl.h"
#include "backend/intel/native/ops.h"
namespace uocr::intel_native {
void expert_topk(const Context& c, const float* logits, std::int64_t rows, std::int64_t experts, std::int64_t k,
                 std::int64_t* ids, float* weights) {
  if (experts < 1 || experts > 128 || k < 1 || k > experts) throw std::invalid_argument("expert topk shape");
  each(c, rows, [=](auto r) {
    float scores[128], peak = -INFINITY, total = 0;
    for (std::int64_t e = 0; e < experts; ++e) peak = sycl::fmax(peak, logits[r * experts + e]);
    for (std::int64_t e = 0; e < experts; ++e) {
      scores[e] = sycl::exp(logits[r * experts + e] - peak);
      total += scores[e];
    }
    for (std::int64_t j = 0; j < k; ++j) {
      std::int64_t best = 0;
      for (std::int64_t e = 1; e < experts; ++e)
        if (scores[e] > scores[best]) best = e;
      ids[r * k + j] = best;
      weights[r * k + j] = scores[best] / total;
      scores[best] = -INFINITY;
    }
  });
}
void expert_ffn(const Context& c, const float* x, std::int64_t rows, std::int64_t hidden, std::int64_t ffn,
                const std::uint16_t* gate, const std::uint16_t* up, const std::uint16_t* down, const std::int64_t* ids,
                const float* weights, std::int64_t k, float* out) {
  Array intermediate(c, DType::kBF16, rows * k * ffn);
  auto* tmp = intermediate.bf16();
  c.impl().kernels.fetch_add(1);
  c.impl().queue.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(rows * k * ffn) * 32, 32),
                              [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                                const auto i = static_cast<std::int64_t>(it.get_group(0)),
                                           lane = static_cast<std::int64_t>(it.get_local_id(0));
                                const auto r = i / (k * ffn), j = (i / ffn) % k, f = i % ffn, e = ids[r * k + j];
                                float g = 0, u = 0;
                                for (std::int64_t h = lane; h < hidden; h += 32) {
                                  const float v = wide(narrow(x[r * hidden + h]));
                                  g += v * wide(gate[(e * ffn + f) * hidden + h]);
                                  u += v * wide(up[(e * ffn + f) * hidden + h]);
                                }
                                g = sycl::reduce_over_group(it.get_sub_group(), g, sycl::plus<float>());
                                u = sycl::reduce_over_group(it.get_sub_group(), u, sycl::plus<float>());
                                if (lane == 0) tmp[i] = narrow(g / (1 + sycl::exp(-g)) * u);
                              });
  c.impl().kernels.fetch_add(1);
  c.impl().queue.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(rows * hidden) * 32, 32),
                              [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                                const auto i = static_cast<std::int64_t>(it.get_group(0)),
                                           lane = static_cast<std::int64_t>(it.get_local_id(0)), r = i / hidden,
                                           h = i % hidden;
                                float total = 0;
                                for (std::int64_t j = 0; j < k; ++j) {
                                  const auto e = ids[r * k + j];
                                  float sum = 0;
                                  for (std::int64_t f = lane; f < ffn; f += 32)
                                    sum += wide(tmp[(r * k + j) * ffn + f]) * wide(down[(e * hidden + h) * ffn + f]);
                                  sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                                  total += weights[r * k + j] * sum;
                                }
                                if (lane == 0) out[i] = total;
                              });
}
void block_summaries(const Context& c, const float* keys, std::int64_t s, std::int64_t d, const float* head,
                     std::int64_t bs, float* out) {
  if (bs < 1 || bs > 64) throw std::invalid_argument("router block size");
  each(c, (s + bs - 1) / bs, [=](auto block) {
    const auto first = block * bs, last = sycl::min(first + bs, s);
    float w[64], peak = -INFINITY, total = 0;
    for (auto t = first; t < last; ++t) {
      float sum = 0;
      for (std::int64_t e = 0; e < d; ++e) sum += keys[t * d + e] * head[e];
      w[t - first] = sycl::clamp(sum / sycl::sqrt(static_cast<float>(d)), -1e4f, 1e4f);
      peak = sycl::fmax(peak, w[t - first]);
    }
    for (auto t = first; t < last; ++t) {
      w[t - first] = sycl::exp(w[t - first] - peak);
      total += w[t - first];
    }
    total = sycl::fmax(total, 1e-6f);
    for (std::int64_t e = 0; e < d; ++e) {
      float sum = 0;
      for (auto t = first; t < last; ++t) sum += w[t - first] / total * keys[t * d + e];
      out[block * d + e] = sum;
    }
  });
}
namespace {
void entmax(float* x, int n) {
  const float peak = x[0];
  for (int i = 0; i < n; ++i) x[i] = (x[i] - peak) / 2;
  float sum = 0, sq = 0, tau[32];
  int support = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
    sq += x[i] * x[i];
    const float rho = static_cast<float>(i + 1), mean = sum / rho;
    tau[i] = mean - sycl::sqrt(sycl::fmax((1 - rho * (sq / rho - mean * mean)) / rho, 0.0f));
    if (tau[i] <= x[i]) ++support;
  }
  const float t = tau[sycl::max(support, 1) - 1];
  for (int i = 0; i < n; ++i) {
    const float v = sycl::fmax(x[i] - t, 0.0f);
    x[i] = v * v;
  }
}
}  // namespace
void route_blocks(const Context& c, const float* z, std::int64_t rows, std::int64_t nblk, std::int64_t first_pos,
                  std::int64_t keys, const std::int64_t* bmin, const std::int64_t* bmax, std::int64_t prompt_blocks,
                  const std::int64_t* query_page, std::int64_t one_page, const RouteParams& params, float* mask) {
  if (params.top_m < 1 || params.top_m > 32 || params.top_k < 1 || params.block_size < 1 || params.sigma <= 0 ||
      params.floor <= 0 || nblk < 1)
    throw std::invalid_argument("router parameters");
  const auto p = params;
  each(c, rows, [=](auto r) {
    const auto pos = first_pos + r, last_remote = pos / p.block_size - p.local_blocks,
               page = query_page ? query_page[r] : one_page;
    bool any = false;
    for (std::int64_t b = 0; b < nblk; ++b)
      any |= b <= last_remote && page >= 0 && b < prompt_blocks && bmin[b] >= 0 && bmin[b] <= page && page <= bmax[b];
    float top[32], bias[32];
    std::int64_t ids[32];
    bool valid[32];
    const int m = static_cast<int>(sycl::min(p.top_m, nblk)),
              k = static_cast<int>(sycl::min(p.top_k, static_cast<std::int64_t>(m)));
    for (int j = 0; j < m; ++j) {
      float best = -INFINITY;
      std::int64_t bi = 0;
      bool bv = false;
      for (std::int64_t b = 0; b < nblk; ++b) {
        bool taken = false;
        for (int t = 0; t < j; ++t) taken |= ids[t] == b;
        if (taken) continue;
        const bool candidate =
            b <= last_remote &&
            (!any || (page >= 0 && b < prompt_blocks && bmin[b] >= 0 && bmin[b] <= page && page <= bmax[b]));
        const float v = candidate ? sycl::clamp(z[r * nblk + b], -1e4f, 1e4f) * p.idx_scale : -1e4f;
        if (v > best) {
          best = v;
          bi = b;
          bv = candidate;
        }
      }
      top[j] = best;
      ids[j] = bi;
      valid[j] = bv;
    }
    entmax(top, m);
    float sum = 0;
    int count = 0;
    for (int j = 0; j < k; ++j) {
      bias[j] = sycl::log(sycl::fmax(top[j], p.floor));
      if (valid[j]) {
        sum += bias[j];
        ++count;
      }
    }
    const float mean = sum / static_cast<float>(sycl::max(count, 1));
    for (int j = 0; j < k; ++j) bias[j] = (bias[j] - mean) / p.sigma;
    for (std::int64_t j = 0; j < keys; ++j) {
      float v = -INFINITY;
      if (j <= pos) {
        if (j > pos - p.local_blocks * p.block_size)
          v = 0;
        else
          for (int t = 0; t < k; ++t)
            if (valid[t] && ids[t] == j / p.block_size) v = bias[t];
      }
      mask[r * keys + j] = v;
    }
  });
}
}  // namespace uocr::intel_native
