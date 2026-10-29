#include "model/vision_encoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "backend/reference/ops.h"

namespace uocr {
namespace ref = reference;
namespace {

// grid rows of `cols` feature vectors each, a newline after every row.
void append_rows_with_newlines(std::vector<float>& out, const float* features, std::int64_t rows, std::int64_t cols,
                               std::int64_t width, const Tensor& newline) {
  for (std::int64_t r = 0; r < rows; ++r) {
    out.insert(out.end(), features + r * cols * width, features + (r + 1) * cols * width);
    out.insert(out.end(), newline.data(), newline.data() + width);
  }
}

}  // namespace

VisionEncoder::VisionEncoder(const WeightSource& weights, const ModelDims& dims)
    : sam_(weights, dims.sam),
      clip_(weights, dims.clip),
      projector_w_(weights.get("model.projector.layers.weight")),
      projector_b_(weights.get("model.projector.layers.bias")),
      newline_(weights.get("model.image_newline")),
      separator_(weights.get("model.view_seperator")) {}

Tensor VisionEncoder::encode_view(const Tensor& image) const {
  const Tensor sam = sam_.forward(image);  // [1024, g, g]
  const Tensor clip = clip_.forward(sam);  // [1 + g * g, 1024]
  const Tensor cells = ref::chw_to_hwc(sam);
  const std::int64_t n = cells.dim(0) * cells.dim(1), cw = clip.dim(1), sw = cells.dim(2);
  Tensor joined({n, cw + sw});
  for (std::int64_t t = 0; t < n; ++t) {
    std::copy_n(clip.row(t + 1), cw, joined.row(t));  // class token dropped
    std::copy_n(cells.data() + t * sw, sw, joined.row(t) + cw);
  }
  return ref::linear(joined, projector_w_, &projector_b_);
}

Tensor VisionEncoder::encode_page(const PageViews& page) const {
  const std::int64_t width = newline_.numel();
  std::vector<float> out;
  const bool tiled = !page.tiles.empty();
  if (tiled) {
    if (static_cast<std::int64_t>(page.tiles.size()) != page.tiles_wide * page.tiles_high) {
      throw std::invalid_argument("encode_page: tile count does not match the tile grid");
    }
    std::vector<Tensor> tiles;
    for (const auto& tile : page.tiles) tiles.push_back(encode_view(tile));
    const auto g = static_cast<std::int64_t>(std::sqrt(static_cast<double>(tiles.front().dim(0))));
    // Tiles are stitched into one (high * g) x (wide * g) grid.
    const std::int64_t cols = page.tiles_wide * g;
    std::vector<float> grid(static_cast<std::size_t>(page.tiles_high * g * cols * width));
    for (std::int64_t ty = 0; ty < page.tiles_high; ++ty)
      for (std::int64_t tx = 0; tx < page.tiles_wide; ++tx) {
        const Tensor& tile = tiles[static_cast<std::size_t>(ty * page.tiles_wide + tx)];
        for (std::int64_t r = 0; r < g; ++r)
          std::copy_n(tile.row(r * g), g * width, grid.data() + ((ty * g + r) * cols + tx * g) * width);
      }
    append_rows_with_newlines(out, grid.data(), page.tiles_high * g, cols, width, newline_);
  }
  const Tensor global = encode_view(page.global);
  const auto g = static_cast<std::int64_t>(std::sqrt(static_cast<double>(global.dim(0))));
  append_rows_with_newlines(out, global.data(), g, g, width, newline_);
  out.insert(out.end(), separator_.data(), separator_.data() + width);
  const auto rows = static_cast<std::int64_t>(out.size()) / width;
  return Tensor({rows, width}, std::move(out));
}

std::int64_t page_image_tokens(std::int64_t global_grid, std::int64_t tile_grid, std::int64_t tiles_wide,
                               std::int64_t tiles_high, bool tiled) {
  std::int64_t tokens = global_grid * (global_grid + 1) + 1;
  if (tiled) {
    tokens += (tile_grid * tiles_wide + 1) * (tile_grid * tiles_high);
  }
  return tokens;
}

}  // namespace uocr
