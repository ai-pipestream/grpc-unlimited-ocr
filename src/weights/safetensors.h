#pragma once

// One safetensors file, memory-mapped read-only. The header is parsed and
// checked once when the file opens; every tensor is then a view into the
// mapping, with no copy.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "tensor/dtype.h"

namespace uocr {

struct TensorInfo {
  DType dtype;
  std::vector<std::int64_t> shape;
  // The tensor's bytes inside the mapping.
  std::span<const std::byte> bytes;

  std::int64_t element_count() const;
};

class SafetensorsFile {
 public:
  // Maps the file and validates its header: the length prefix, the JSON,
  // every dtype, every shape, and that the data offsets tile the data
  // section exactly (in bounds, no overlap, no gap). Throws
  // std::runtime_error naming the file and the first problem found.
  explicit SafetensorsFile(const std::filesystem::path& path);
  ~SafetensorsFile();

  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  SafetensorsFile(SafetensorsFile&& other) noexcept;
  SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;

  const std::filesystem::path& path() const { return path_; }
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }

  // Nullptr when the file holds no tensor of that name.
  const TensorInfo* find(const std::string& name) const;

 private:
  void release() noexcept;

  std::filesystem::path path_;
  const std::byte* mapping_ = nullptr;
  std::size_t mapping_size_ = 0;
  std::map<std::string, TensorInfo> tensors_;
};

// Parses and validates a header against a data section of `data_size`
// bytes that starts at `data`. Exposed for tests; SafetensorsFile uses it.
std::map<std::string, TensorInfo> parse_safetensors_header(std::string_view header_json,
                                                           const std::byte* data,
                                                           std::size_t data_size);

}  // namespace uocr
