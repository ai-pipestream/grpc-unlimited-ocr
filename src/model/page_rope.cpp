#include "model/page_rope.h"

#include <cmath>
#include <stdexcept>

namespace uocr {

PageRope::PageRope(std::int64_t dim, double tok_theta, double page_theta, std::int64_t page_pairs,
                   std::int64_t page_mod)
    : dim_(dim), page_mod_(page_mod) {
  const std::int64_t half = dim / 2;
  const std::int64_t n_tok = half - page_pairs;
  if (page_pairs <= 0 || n_tok <= 0) {
    throw std::invalid_argument("PageRope: page_pairs must leave room for token pairs");
  }
  // "resample": the token axis spans the whole spectrum at its reduced pair count.
  for (std::int64_t i = 0; i < n_tok; ++i) {
    inv_tok_.push_back(static_cast<float>(1.0 / std::pow(tok_theta, static_cast<double>(i) / static_cast<double>(n_tok))));
  }
  for (std::int64_t i = 0; i < page_pairs; ++i) {
    inv_page_.push_back(
        static_cast<float>(1.0 / std::pow(page_theta, static_cast<double>(i) / static_cast<double>(page_pairs))));
  }
}

void PageRope::apply(Tensor& x, const std::vector<RopeCoord>& coords) const {
  if (static_cast<std::int64_t>(coords.size()) != x.rows() || x.shape().back() % dim_ != 0) {
    throw std::invalid_argument("PageRope::apply: one coordinate per row, width a multiple of dim");
  }
  const std::int64_t half = dim_ / 2, heads = x.shape().back() / dim_;
  std::vector<float> cos_v(static_cast<std::size_t>(half)), sin_v(static_cast<std::size_t>(half));
  for (std::int64_t r = 0; r < x.rows(); ++r) {
    const auto& c = coords[static_cast<std::size_t>(r)];
    const auto t = static_cast<float>(c.tok);
    const auto p = static_cast<float>(page_mod_ > 0 ? ((c.page % page_mod_) + page_mod_) % page_mod_ : c.page);
    for (std::size_t i = 0; i < inv_tok_.size(); ++i) {
      const float a = t * inv_tok_[i];  // fp32 angle, as upstream computes it
      cos_v[i] = std::cos(a);
      sin_v[i] = std::sin(a);
    }
    for (std::size_t i = 0; i < inv_page_.size(); ++i) {
      const float a = p * inv_page_[i];
      cos_v[inv_tok_.size() + i] = std::cos(a);
      sin_v[inv_tok_.size() + i] = std::sin(a);
    }
    float* row = x.row(r);
    for (std::int64_t h = 0; h < heads; ++h) {
      float* v = row + h * dim_;
      for (std::int64_t i = 0; i < half; ++i) {
        const float a = v[i], b = v[i + half];
        v[i] = a * cos_v[static_cast<std::size_t>(i)] - b * sin_v[static_cast<std::size_t>(i)];
        v[i + half] = b * cos_v[static_cast<std::size_t>(i)] + a * sin_v[static_cast<std::size_t>(i)];
      }
    }
  }
}

}  // namespace uocr
