// bf16 conversions and the dtype table.

#include <cmath>
#include <limits>

#include "check.h"
#include "tensor/dtype.h"

using namespace uocr;

int main() {
  return run_tests([] {
    require(element_size(DType::kBF16) == 2 && element_size(DType::kF32) == 4 && element_size(DType::kI64) == 8,
            "element sizes");
    for (const auto dtype : {DType::kBF16, DType::kF16, DType::kF32, DType::kI64, DType::kI32, DType::kU8}) {
      require(parse_dtype(dtype_name(dtype)) == dtype, "dtype name round trip");
    }
    require(!parse_dtype("F64").has_value(), "F64 is not stored");

    // Values exactly representable in bf16 survive both ways.
    for (const float value : {0.0f, -0.0f, 1.0f, -2.5f, 1024.0f, 3.0517578e-05f}) {
      require(bf16_to_f32(f32_to_bf16(value)) == value, "exact bf16 round trip");
    }
    // Round to nearest even: 1 + 2^-8 is a tie between 1 and 1 + 2^-7.
    require(bf16_to_f32(f32_to_bf16(1.0f + 0x1p-8f)) == 1.0f, "tie rounds to even (down)");
    require(bf16_to_f32(f32_to_bf16(1.0f + 3 * 0x1p-8f)) == 1.0f + 0x1p-6f, "tie rounds to even (up)");
    require(std::isinf(bf16_to_f32(f32_to_bf16(std::numeric_limits<float>::max()))), "max float overflows to inf");
    require(std::isnan(bf16_to_f32(f32_to_bf16(std::numeric_limits<float>::quiet_NaN()))), "NaN stays NaN");
  });
}
