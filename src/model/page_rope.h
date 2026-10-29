#pragma once

// The decoder's two-axis rotary embedding. Each position carries a
// (token, page) coordinate; of the 64 frequency pairs of a 128-dim head,
// 63 rotate with the token coordinate (theta 1e4, resampled across the
// whole spectrum) and one with the page coordinate folded mod 2 (theta
// 256), so attention sees "same page / neighbouring page" rather than a
// page number. The main attention and the block router use the same table.

#include <cstdint>
#include <vector>

#include "tensor/tensor.h"

namespace uocr {

struct RopeCoord {
  std::int64_t tok = 0;
  std::int64_t page = 0;
};

class PageRope {
 public:
  explicit PageRope(std::int64_t dim = 128, double tok_theta = 1e4, double page_theta = 256.0,
                    std::int64_t page_pairs = 1, std::int64_t page_mod = 2);

  // Rotates each head of x [L, heads * dim] in place at the given
  // coordinates (one per row), rotate-half convention.
  void apply(Tensor& x, const std::vector<RopeCoord>& coords) const;

  std::int64_t dim() const { return dim_; }

 private:
  std::int64_t dim_;
  std::int64_t page_mod_;
  std::vector<float> inv_tok_;
  std::vector<float> inv_page_;
};

}  // namespace uocr
