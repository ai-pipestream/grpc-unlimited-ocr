#pragma once

// Element types the engine stores and the conversions between them.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace uocr {

enum class DType : std::uint8_t {
  kBF16,
  kF16,
  kF32,
  kI64,
  kI32,
  kU8,
};

// Bytes per element.
std::size_t element_size(DType dtype);

// The safetensors spelling ("BF16", "F32", ...).
std::string_view dtype_name(DType dtype);

// Parses a safetensors dtype spelling; nullopt for types the engine does
// not store.
std::optional<DType> parse_dtype(std::string_view name);

// bf16 is the top half of an IEEE-754 binary32.
inline float bf16_to_f32(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

// Round to nearest even, NaN kept quiet. Matches the conversion PyTorch
// applies when it casts float32 to bfloat16.
inline std::uint16_t f32_to_bf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
  }
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

}  // namespace uocr
