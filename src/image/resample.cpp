#include "image/resample.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uocr {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;

double bicubic(double x) {
  constexpr double a = -0.5;
  x = std::fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

struct Taps {
  std::int64_t first = 0;
  std::vector<int> coeffs;  // fixed point, kPrecisionBits
};

// Pillow's precompute_coeffs plus normalize_coeffs_8bpc for one axis.
std::vector<Taps> taps(std::int64_t in, std::int64_t out) {
  const double scale = static_cast<double>(in) / static_cast<double>(out);
  const double filterscale = std::max(scale, 1.0);
  const double support = 2.0 * filterscale;
  const double ss = 1.0 / filterscale;
  std::vector<Taps> all(static_cast<std::size_t>(out));
  for (std::int64_t xx = 0; xx < out; ++xx) {
    const double center = (static_cast<double>(xx) + 0.5) * scale;
    auto xmin = static_cast<std::int64_t>(center - support + 0.5);
    xmin = std::max<std::int64_t>(xmin, 0);
    auto xmax = static_cast<std::int64_t>(center + support + 0.5);
    xmax = std::min(xmax, in) - xmin;
    std::vector<double> k(static_cast<std::size_t>(xmax));
    double ww = 0.0;
    for (std::int64_t x = 0; x < xmax; ++x) {
      const double w = bicubic((static_cast<double>(x + xmin) - center + 0.5) * ss);
      k[static_cast<std::size_t>(x)] = w;
      ww += w;
    }
    auto& t = all[static_cast<std::size_t>(xx)];
    t.first = xmin;
    for (double w : k) {
      if (ww != 0.0) w /= ww;
      const double scaled = w * static_cast<double>(1 << kPrecisionBits);
      t.coeffs.push_back(static_cast<int>(w < 0 ? -0.5 + scaled : 0.5 + scaled));
    }
  }
  return all;
}

std::uint8_t clip8(std::int64_t acc) {
  return static_cast<std::uint8_t>(std::clamp<std::int64_t>(acc >> kPrecisionBits, 0, 255));
}

RgbImage horizontal(const RgbImage& src, std::int64_t width) {
  const auto t = taps(src.width, width);
  RgbImage out = blank_image(width, src.height, 0);
  for (std::int64_t y = 0; y < src.height; ++y) {
    for (std::int64_t x = 0; x < width; ++x) {
      const auto& tap = t[static_cast<std::size_t>(x)];
      std::int64_t acc[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
      for (std::size_t k = 0; k < tap.coeffs.size(); ++k) {
        const std::uint8_t* p = src.at(tap.first + static_cast<std::int64_t>(k), y);
        for (int c = 0; c < 3; ++c) acc[c] += static_cast<std::int64_t>(p[c]) * tap.coeffs[k];
      }
      for (int c = 0; c < 3; ++c) out.at(x, y)[c] = clip8(acc[c]);
    }
  }
  return out;
}

RgbImage vertical(const RgbImage& src, std::int64_t height) {
  const auto t = taps(src.height, height);
  RgbImage out = blank_image(src.width, height, 0);
  for (std::int64_t y = 0; y < height; ++y) {
    const auto& tap = t[static_cast<std::size_t>(y)];
    for (std::int64_t x = 0; x < src.width; ++x) {
      std::int64_t acc[3] = {1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1), 1 << (kPrecisionBits - 1)};
      for (std::size_t k = 0; k < tap.coeffs.size(); ++k) {
        const std::uint8_t* p = src.at(x, tap.first + static_cast<std::int64_t>(k));
        for (int c = 0; c < 3; ++c) acc[c] += static_cast<std::int64_t>(p[c]) * tap.coeffs[k];
      }
      for (int c = 0; c < 3; ++c) out.at(x, y)[c] = clip8(acc[c]);
    }
  }
  return out;
}

// Python's round(): half to even.
std::int64_t round_half_even(double v) { return static_cast<std::int64_t>(std::nearbyint(v)); }

}  // namespace

RgbImage blank_image(std::int64_t width, std::int64_t height, std::uint8_t fill) {
  return RgbImage{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width * height * 3), fill)};
}

RgbImage resize_bicubic(const RgbImage& src, std::int64_t width, std::int64_t height) {
  if (width <= 0 || height <= 0 || src.width <= 0 || src.height <= 0) {
    throw std::invalid_argument("resize_bicubic: empty image or target");
  }
  if (width == src.width && height == src.height) return src;
  RgbImage tmp = width != src.width ? horizontal(src, width) : src;
  return height != tmp.height ? vertical(tmp, height) : tmp;
}

RgbImage letterbox(const RgbImage& src, std::int64_t size, std::uint8_t fill) {
  if (src.width == src.height) return resize_bicubic(src, size, size);
  RgbImage out = blank_image(size, size, fill);
  RgbImage scaled;
  std::int64_t ox = 0, oy = 0;
  if (src.width > src.height) {
    const std::int64_t h = round_half_even(static_cast<double>(src.height) / static_cast<double>(src.width) * static_cast<double>(size));
    scaled = resize_bicubic(src, size, h);
    oy = round_half_even(static_cast<double>(size - h) * 0.5);
  } else {
    const std::int64_t w = round_half_even(static_cast<double>(src.width) / static_cast<double>(src.height) * static_cast<double>(size));
    scaled = resize_bicubic(src, w, size);
    ox = round_half_even(static_cast<double>(size - w) * 0.5);
  }
  for (std::int64_t y = 0; y < scaled.height; ++y) {
    std::copy_n(scaled.at(0, y), scaled.width * 3, out.at(ox, oy + y));
  }
  return out;
}

RgbImage crop(const RgbImage& src, std::int64_t x, std::int64_t y, std::int64_t w, std::int64_t h) {
  if (x < 0 || y < 0 || x + w > src.width || y + h > src.height) {
    throw std::invalid_argument("crop: box outside the image");
  }
  RgbImage out = blank_image(w, h, 0);
  for (std::int64_t r = 0; r < h; ++r) std::copy_n(src.at(x, y + r), w * 3, out.at(0, r));
  return out;
}

}  // namespace uocr
