#include "image/page_preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "tensor/dtype.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include <stb_image.h>

namespace uocr {

RgbImage decode_image(std::span<const std::uint8_t> bytes) {
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("image too large to decode");
  }
  int w = 0, h = 0, channels = 0;
  stbi_uc* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &channels, 3);
  if (data == nullptr) {
    throw std::invalid_argument(std::string("not a decodable PNG or JPEG: ") + stbi_failure_reason());
  }
  RgbImage out{w, h, std::vector<std::uint8_t>(data, data + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3)};
  stbi_image_free(data);
  return out;
}

std::pair<std::int64_t, std::int64_t> tile_grid(std::int64_t width, std::int64_t height, const PreprocessOptions& o) {
  // Candidate grids ordered by tile count, as the reference sorts them.
  std::set<std::pair<std::int64_t, std::int64_t>> unique;
  for (std::int64_t n = o.min_tiles; n <= o.max_tiles; ++n)
    for (std::int64_t i = 1; i <= n; ++i)
      for (std::int64_t j = 1; j <= n; ++j)
        if (i * j <= o.max_tiles && i * j >= o.min_tiles) unique.insert({i, j});
  std::vector<std::pair<std::int64_t, std::int64_t>> ratios(unique.begin(), unique.end());
  std::ranges::stable_sort(ratios, {}, [](const auto& r) { return r.first * r.second; });
  const double aspect = static_cast<double>(width) / static_cast<double>(height);
  const double area = static_cast<double>(width * height);
  double best_diff = std::numeric_limits<double>::infinity();
  std::pair<std::int64_t, std::int64_t> best{1, 1};
  for (const auto& r : ratios) {
    const double diff = std::fabs(aspect - static_cast<double>(r.first) / static_cast<double>(r.second));
    if (diff < best_diff) {
      best_diff = diff;
      best = r;
    } else if (diff == best_diff &&
               area > 0.5 * static_cast<double>(o.tile_size * o.tile_size * r.first * r.second)) {
      best = r;
    }
  }
  return best;
}

Tensor to_model_tensor(const RgbImage& image, bool round_through_bf16) {
  Tensor out({3, image.height, image.width});
  const std::int64_t plane = image.width * image.height;
  for (std::int64_t p = 0; p < plane; ++p) {
    for (int c = 0; c < 3; ++c) {
      float v = (static_cast<float>(image.pixels[static_cast<std::size_t>(p * 3 + c)]) / 255.0f - 0.5f) / 0.5f;
      if (round_through_bf16) v = bf16_to_f32(f32_to_bf16(v));
      out.data()[c * plane + p] = v;
    }
  }
  return out;
}

PageViews preprocess_page(const RgbImage& page, const PreprocessOptions& o) {
  constexpr std::uint8_t kPad = 127;  // int(0.5 * 255)
  PageViews views;
  views.global = to_model_tensor(letterbox(page, o.base_size, kPad), o.round_through_bf16);
  const bool small = page.width <= o.tile_size && page.height <= o.tile_size;
  if (!o.tiled || small) return views;
  const auto [wide, high] = tile_grid(page.width, page.height, o);
  const RgbImage resized = resize_bicubic(page, o.tile_size * wide, o.tile_size * high);
  for (std::int64_t i = 0; i < wide * high; ++i) {
    const std::int64_t x = (i % wide) * o.tile_size, y = (i / wide) * o.tile_size;
    views.tiles.push_back(to_model_tensor(crop(resized, x, y, o.tile_size, o.tile_size), o.round_through_bf16));
  }
  views.tiles_wide = wide;
  views.tiles_high = high;
  return views;
}

}  // namespace uocr
