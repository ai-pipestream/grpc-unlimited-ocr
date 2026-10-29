#pragma once

// One page's image tokens: SAM, then CLIP over SAM's features, the two
// concatenated per cell and projected to the decoder width, then laid out
// with a newline embedding after each row and a separator after the page.

#include "model/clip_encoder.h"
#include "model/sam_encoder.h"
#include "model/weight_source.h"
#include "tensor/tensor.h"

namespace uocr {

// A page as the model sees it: one global view, and in tiled mode a grid of
// tiles, row-major, tiles_wide x tiles_high.
struct PageViews {
  Tensor global;              // [3, 1024, 1024]
  std::vector<Tensor> tiles;  // each [3, 640, 640]; empty in base mode
  std::int64_t tiles_wide = 1;
  std::int64_t tiles_high = 1;
};

class VisionEncoder {
 public:
  explicit VisionEncoder(const WeightSource& weights, const ModelDims& dims = {});

  // Projected features of one view: [g * g, hidden].
  Tensor encode_view(const Tensor& image) const;

  // The page's image-token embeddings in prompt order: tile rows (each
  // followed by the newline), then the global rows (each followed by the
  // newline), then the separator. Base mode: 16 x 17 + 1 = 273 rows.
  Tensor encode_page(const PageViews& page) const;

 private:
  SamEncoder sam_;
  ClipEncoder clip_;
  Tensor projector_w_, projector_b_, newline_, separator_;
};

// Image tokens a page contributes to the prompt; must equal
// encode_page(...).dim(0).
std::int64_t page_image_tokens(std::int64_t global_grid, std::int64_t tile_grid, std::int64_t tiles_wide,
                               std::int64_t tiles_high, bool tiled);

}  // namespace uocr
