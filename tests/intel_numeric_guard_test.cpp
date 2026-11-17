#include <array>

#include "intel_numeric_checks.h"
using namespace uocr::intel_testing;
int main() {
  return run_tests([] {
    const std::array<float, 2> finite{1, 2};
    require(finite_relative_error(finite, finite) == 0, "finite equality");
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
      const std::array<float, 2> corrupt{1, value};
      require_throws([&] { finite_relative_error(corrupt, finite); }, "nonfinite candidate mutation rejected");
      require_throws([&] { finite_relative_error(finite, corrupt); }, "nonfinite reference mutation rejected");
      require_throws([&] { mask_absolute_error(corrupt, finite); }, "nonfinite unmasked mutation rejected");
    }
    const std::array<float, 1> short_input{1};
    require_throws([&] { finite_relative_error(short_input, finite); }, "short candidate rejected");
    require_throws([&] { finite_relative_error(finite, short_input); }, "short reference rejected");
    require_throws([&] { mask_absolute_error(short_input, finite); }, "short mask rejected");
    const std::array<float, 2> mask{1, -std::numeric_limits<float>::infinity()};
    require(mask_absolute_error(mask, mask) == 0, "negative infinity mask sentinel accepted");
    const std::array<float, 2> positive{1, std::numeric_limits<float>::infinity()},
        nan{1, std::numeric_limits<float>::quiet_NaN()};
    require_throws([&] { mask_absolute_error(positive, mask); }, "wrong infinity sign rejected");
    require_throws([&] { mask_absolute_error(nan, mask); }, "NaN masked mutation rejected");
    require_throws([&] { mask_absolute_error(positive, positive); }, "infinite reference is not finite score");
  });
}
