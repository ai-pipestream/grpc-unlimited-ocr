#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

#include "check.h"
namespace uocr::intel_testing {
inline double finite_relative_error(std::span<const float> got, std::span<const float> want) {
  require(got.size() == want.size() && !want.empty(), "numeric comparison size equality and nonempty reference");
  double worst = 0, scale = 1e-12;
  for (std::size_t i = 0; i < want.size(); ++i) {
    require(std::isfinite(got[i]) && std::isfinite(want[i]),
            "numeric comparison requires finite candidate and reference");
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
  }
  return worst / scale;
}
inline double mask_absolute_error(std::span<const float> got, std::span<const float> want) {
  require(got.size() == want.size() && !want.empty(), "mask comparison size equality and nonempty reference");
  double worst = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i] == -std::numeric_limits<float>::infinity()) {
      require(got[i] == want[i], "masked keys must be negative infinity");
      continue;
    }
    require(std::isfinite(want[i]) && std::isfinite(got[i]), "unmasked scores must be finite");
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
  }
  return worst;
}
}  // namespace uocr::intel_testing
