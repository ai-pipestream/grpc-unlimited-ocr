#pragma once

// 8-bit RGB images and the resampling the model's preprocessing used:
// Pillow's antialiased bicubic resize (fixed-point, horizontal then vertical
// pass, each rounded back to 8 bits), reproduced exactly so the pixels the
// vision encoder sees match the ones the model was trained and evaluated on.

#include <cstdint>
#include <vector>

namespace uocr {

struct RgbImage {
  std::int64_t width = 0;
  std::int64_t height = 0;
  std::vector<std::uint8_t> pixels;  // row-major RGB

  std::uint8_t* at(std::int64_t x, std::int64_t y) { return pixels.data() + (y * width + x) * 3; }
  const std::uint8_t* at(std::int64_t x, std::int64_t y) const { return pixels.data() + (y * width + x) * 3; }
};

RgbImage blank_image(std::int64_t width, std::int64_t height, std::uint8_t fill);

// Image.resize((width, height), Image.BICUBIC).
RgbImage resize_bicubic(const RgbImage& src, std::int64_t width, std::int64_t height);

// ImageOps.pad(image, (size, size), BICUBIC, color=(fill, fill, fill)): fit
// inside the square keeping the aspect ratio, centred on `fill`.
RgbImage letterbox(const RgbImage& src, std::int64_t size, std::uint8_t fill);

// Image.crop((x, y, x + w, y + h)) inside the image.
RgbImage crop(const RgbImage& src, std::int64_t x, std::int64_t y, std::int64_t w, std::int64_t h);

}  // namespace uocr
