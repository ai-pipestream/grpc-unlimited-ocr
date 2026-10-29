#pragma once

// A dense, row-major, host-resident fp32 tensor: the reference backend's
// only data type. Fast backends have their own device buffers; this one
// favours being obviously correct.

#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uocr {

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<std::int64_t> shape, float fill = 0.0f)
      : shape_(std::move(shape)), data_(static_cast<std::size_t>(count(shape_)), fill) {}
  Tensor(std::vector<std::int64_t> shape, std::vector<float> data) : shape_(std::move(shape)), data_(std::move(data)) {
    if (static_cast<std::int64_t>(data_.size()) != count(shape_)) {
      throw std::invalid_argument("tensor data size does not match its shape");
    }
  }

  const std::vector<std::int64_t>& shape() const { return shape_; }
  std::int64_t dim(std::size_t axis) const { return shape_.at(axis); }
  std::size_t rank() const { return shape_.size(); }
  std::int64_t numel() const { return static_cast<std::int64_t>(data_.size()); }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
  std::span<float> span() { return data_; }
  std::span<const float> span() const { return data_; }

  // Row `index` of a tensor viewed as [numel / last, last].
  float* row(std::int64_t index) { return data_.data() + index * shape_.back(); }
  const float* row(std::int64_t index) const { return data_.data() + index * shape_.back(); }
  std::int64_t rows() const { return shape_.empty() ? 0 : numel() / shape_.back(); }

  // Same data, new shape of the same element count.
  Tensor reshaped(std::vector<std::int64_t> shape) && {
    if (count(shape) != numel()) {
      throw std::invalid_argument("reshape changes the element count");
    }
    shape_ = std::move(shape);
    return std::move(*this);
  }

  static std::int64_t count(const std::vector<std::int64_t>& shape) {
    std::int64_t n = 1;
    for (const std::int64_t d : shape) {
      n *= d;
    }
    return n;
  }

 private:
  std::vector<std::int64_t> shape_;
  std::vector<float> data_;
};

}  // namespace uocr
