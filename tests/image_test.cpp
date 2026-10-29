// Page preprocessing: resize properties, letterbox geometry, tile grids and
// the pixel scale.

#include <cmath>

#include "check.h"
#include "image/page_preprocess.h"

using namespace uocr;

namespace {

RgbImage gradient(std::int64_t w, std::int64_t h) {
  RgbImage img = blank_image(w, h, 0);
  for (std::int64_t y = 0; y < h; ++y)
    for (std::int64_t x = 0; x < w; ++x) {
      img.at(x, y)[0] = static_cast<std::uint8_t>((x * 255) / std::max<std::int64_t>(w - 1, 1));
      img.at(x, y)[1] = static_cast<std::uint8_t>((y * 255) / std::max<std::int64_t>(h - 1, 1));
      img.at(x, y)[2] = 200;
    }
  return img;
}

}  // namespace

int main() {
  return run_tests([] {
    const RgbImage flat = blank_image(37, 23, 90);
    for (const auto v : resize_bicubic(flat, 64, 64).pixels) require(v == 90, "bicubic keeps a flat image flat");
    for (const auto v : resize_bicubic(flat, 11, 7).pixels) require(v == 90, "downscale keeps a flat image flat");
    const RgbImage g = gradient(50, 40);
    require(resize_bicubic(g, 50, 40).pixels == g.pixels, "same-size resize is a copy");
    const RgbImage up = resize_bicubic(g, 100, 40);
    require(up.width == 100 && up.at(0, 0)[2] == 200 && up.at(99, 39)[2] == 200, "upscale geometry");

    // A portrait page: 1240 x 1754 fits as 724 x 1024, centred 150 px from the left.
    const RgbImage page = blank_image(1240, 1754, 255);
    const RgbImage boxed = letterbox(page, 1024, 127);
    require(boxed.width == 1024 && boxed.height == 1024, "letterbox size");
    require(boxed.at(149, 500)[0] == 127 && boxed.at(150, 500)[0] == 255 && boxed.at(873, 500)[0] == 255 &&
                boxed.at(874, 500)[0] == 127,
            "portrait page placed at x 150..873");
    const RgbImage wide = letterbox(blank_image(2000, 1000, 10), 1024, 127);
    require(wide.at(500, 255)[0] == 127 && wide.at(500, 256)[0] == 10 && wide.at(500, 767)[0] == 10 &&
                wide.at(500, 768)[0] == 127,
            "landscape page placed at y 256..767");

    PreprocessOptions o;
    require(tile_grid(1240, 1754, o) == std::pair<std::int64_t, std::int64_t>{2, 3}, "A4 portrait tiles 2 x 3");
    // 2:1 matches 2 x 1 and 4 x 2 equally; a page with more than half of 4 x 2's
    // pixel area takes the bigger grid.
    require(tile_grid(2000, 1000, o) == std::pair<std::int64_t, std::int64_t>{4, 2}, "large 2:1 page tiles 4 x 2");
    require(tile_grid(1000, 500, o) == std::pair<std::int64_t, std::int64_t>{2, 1}, "small 2:1 page tiles 2 x 1");
    require(tile_grid(1000, 1000, o) == std::pair<std::int64_t, std::int64_t>{2, 2}, "a large square tiles 2 x 2");

    const PageViews base = preprocess_page(page, o);
    require(base.global.shape() == std::vector<std::int64_t>{3, 1024, 1024} && base.tiles.empty(), "base mode views");
    o.tiled = true;
    const PageViews tiled = preprocess_page(page, o);
    require(tiled.tiles.size() == 6 && tiled.tiles_wide == 2 && tiled.tiles_high == 3, "tiled views");
    require(preprocess_page(blank_image(600, 400, 0), o).tiles.empty(), "a page within one tile is not tiled");

    const Tensor t = to_model_tensor(blank_image(1, 1, 255), true);
    require(t.data()[0] == 1.0f, "255 maps to 1");
    const Tensor mid = to_model_tensor(blank_image(1, 1, 127), true);
    require(std::fabs(mid.data()[0] + 0.00392f) < 2e-4f, "127 maps just below 0, through bf16");

    require_throws([] { decode_image(std::vector<std::uint8_t>{1, 2, 3}); }, "garbage is not an image");
  });
}
