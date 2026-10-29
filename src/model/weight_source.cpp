#include "model/weight_source.h"

#include <cstring>
#include <format>
#include <stdexcept>

#include "tensor/dtype.h"
#include "weights/sharded_weights.h"

namespace uocr {

Tensor CheckpointWeights::get(const std::string& name) const {
  const TensorInfo& info = weights_.at(name);
  std::vector<float> values(static_cast<std::size_t>(info.element_count()));
  switch (info.dtype) {
    case DType::kBF16: {
      const auto* raw = info.bytes.data();
      for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint16_t bits = 0;
        std::memcpy(&bits, raw + 2 * i, 2);
        values[i] = bf16_to_f32(bits);
      }
      break;
    }
    case DType::kF32:
      std::memcpy(values.data(), info.bytes.data(), info.bytes.size());
      break;
    default:
      throw std::invalid_argument(std::format("{}: {} is not a float parameter", name, dtype_name(info.dtype)));
  }
  return Tensor(info.shape, std::move(values));
}

Tensor TableWeights::get(const std::string& name) const {
  const auto it = table_.find(name);
  if (it == table_.end()) {
    throw std::out_of_range(std::format("no tensor named {}", name));
  }
  return it->second;
}

}  // namespace uocr
