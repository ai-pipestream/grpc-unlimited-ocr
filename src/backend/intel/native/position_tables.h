#pragma once
#include "backend/intel/native/context.h"
#include "tensor/tensor.h"
namespace uocr::intel_native {
// Learned table interpolation executes on the GPU; host builds only shape-dependent filter taps.
Array resize_position_grid(const Context&, const Tensor&, std::int64_t source_grid, std::int64_t target_grid,
                           std::int64_t width, bool class_token);
Array relative_position_table(const Context&, const Tensor&, std::int64_t span);
}  // namespace uocr::intel_native
