#pragma once

// From a decoded page to the tensors the vision encoder takes: a 1024
// letterboxed global view, and in tiled mode a grid of 640 tiles chosen by
// the page's aspect ratio. Pixels become (v / 255 - 0.5) / 0.5, rounded
// through bf16 as the model's own preprocessing does.

#include <cstdint>
#include <span>
#include <string>
#include <utility>

#include "image/resample.h"
#include "model/vision_encoder.h"

namespace uocr {

struct PreprocessOptions {
  bool tiled = false;  // false = base mode (one global view)
  std::int64_t base_size = 1024;
  std::int64_t tile_size = 640;
  std::int64_t min_tiles = 2;
  std::int64_t max_tiles = 16;
  bool round_through_bf16 = true;
};

// Decodes PNG or JPEG bytes into RGB (alpha dropped, grey expanded).
// Throws std::invalid_argument when the bytes are not a supported image.
RgbImage decode_image(std::span<const std::uint8_t> bytes);

// The tile grid (wide, high) for a page of this size, as the model's
// preprocessing picks it.
std::pair<std::int64_t, std::int64_t> tile_grid(std::int64_t width, std::int64_t height, const PreprocessOptions& options);

PageViews preprocess_page(const RgbImage& page, const PreprocessOptions& options = {});

// [3, H, W] in model scale.
Tensor to_model_tensor(const RgbImage& image, bool round_through_bf16);

}  // namespace uocr
