#pragma once
#include "model/intel/device_params.h"
namespace uocr::intel_native {
enum class WeightFormat : std::uint8_t { kBf16 };
struct DeviceWeight {
  Array data;
  std::int64_t rows = 0, cols = 0;
};
DeviceWeight device_weight(const Context&, const Tensor&, WeightFormat);
DeviceWeight stacked_weight(const Context&, const WeightSource&, const std::vector<std::string>&, WeightFormat);
void apply_weight(const Context&, const float*, std::int64_t, const DeviceWeight&, const float*, float*,
                  std::int64_t = 0, std::int64_t = -1);
}  // namespace uocr::intel_native
