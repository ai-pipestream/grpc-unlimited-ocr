#include "tensor/dtype.h"

#include <array>
#include <utility>

namespace uocr {
namespace {

struct DTypeRow {
  DType dtype;
  std::string_view name;
  std::size_t size;
};

constexpr std::array<DTypeRow, 6> kRows{{
    {DType::kBF16, "BF16", 2},
    {DType::kF16, "F16", 2},
    {DType::kF32, "F32", 4},
    {DType::kI64, "I64", 8},
    {DType::kI32, "I32", 4},
    {DType::kU8, "U8", 1},
}};

const DTypeRow& row(DType dtype) {
  for (const auto& entry : kRows) {
    if (entry.dtype == dtype) {
      return entry;
    }
  }
  std::unreachable();
}

}  // namespace

std::size_t element_size(DType dtype) { return row(dtype).size; }

std::string_view dtype_name(DType dtype) { return row(dtype).name; }

std::optional<DType> parse_dtype(std::string_view name) {
  for (const auto& entry : kRows) {
    if (entry.name == name) {
      return entry.dtype;
    }
  }
  return std::nullopt;
}

}  // namespace uocr
