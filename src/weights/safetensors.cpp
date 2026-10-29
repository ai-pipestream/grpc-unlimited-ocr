#include "weights/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace uocr {
namespace {

// Larger headers are not something any real checkpoint writes; the cap
// keeps a corrupt length prefix from asking for gigabytes of JSON.
constexpr std::uint64_t kMaxHeaderBytes = 100ULL * 1024 * 1024;

struct Extent {
  std::uint64_t begin;
  std::uint64_t end;
  const std::string* name;
};

std::vector<std::int64_t> parse_shape(const nlohmann::json& value, const std::string& name) {
  if (!value.is_array()) {
    throw std::runtime_error(std::format("tensor {}: shape is not an array", name));
  }
  std::vector<std::int64_t> shape;
  shape.reserve(value.size());
  for (const auto& dim : value) {
    if (!dim.is_number_integer() || dim.get<std::int64_t>() < 0) {
      throw std::runtime_error(std::format("tensor {}: shape holds a non-integer or negative dim", name));
    }
    shape.push_back(dim.get<std::int64_t>());
  }
  return shape;
}

std::uint64_t checked_byte_count(const std::vector<std::int64_t>& shape, DType dtype,
                                 const std::string& name) {
  std::uint64_t count = element_size(dtype);
  for (const std::int64_t dim : shape) {
    const auto udim = static_cast<std::uint64_t>(dim);
    if (udim != 0 && count > std::numeric_limits<std::uint64_t>::max() / udim) {
      throw std::runtime_error(std::format("tensor {}: shape overflows", name));
    }
    count *= udim;
  }
  return count;
}

}  // namespace

std::int64_t TensorInfo::element_count() const {
  std::int64_t count = 1;
  for (const std::int64_t dim : shape) {
    count *= dim;
  }
  return count;
}

std::map<std::string, TensorInfo> parse_safetensors_header(std::string_view header_json,
                                                           const std::byte* data,
                                                           std::size_t data_size) {
  const auto header = nlohmann::json::parse(header_json, nullptr, /*allow_exceptions=*/false);
  if (header.is_discarded() || !header.is_object()) {
    throw std::runtime_error("header is not a JSON object");
  }
  std::map<std::string, TensorInfo> tensors;
  std::vector<Extent> extents;
  for (const auto& [name, entry] : header.items()) {
    if (name == "__metadata__") {
      continue;
    }
    if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
        !entry.contains("data_offsets")) {
      throw std::runtime_error(std::format("tensor {}: entry lacks dtype, shape or data_offsets", name));
    }
    const auto& dtype_value = entry["dtype"];
    const auto dtype = dtype_value.is_string() ? parse_dtype(dtype_value.get<std::string>()) : std::nullopt;
    if (!dtype) {
      throw std::runtime_error(std::format("tensor {}: unsupported dtype {}", name, dtype_value.dump()));
    }
    auto shape = parse_shape(entry["shape"], name);
    const auto& offsets = entry["data_offsets"];
    if (!offsets.is_array() || offsets.size() != 2 || !offsets[0].is_number_unsigned() ||
        !offsets[1].is_number_unsigned()) {
      throw std::runtime_error(std::format("tensor {}: data_offsets is not two unsigned integers", name));
    }
    const auto begin = offsets[0].get<std::uint64_t>();
    const auto end = offsets[1].get<std::uint64_t>();
    if (begin > end || end > data_size) {
      throw std::runtime_error(std::format("tensor {}: data_offsets [{}, {}) outside a {}-byte data section",
                                           name, begin, end, data_size));
    }
    if (end - begin != checked_byte_count(shape, *dtype, name)) {
      throw std::runtime_error(std::format("tensor {}: {} bytes stored, shape and dtype need {}", name,
                                           end - begin, checked_byte_count(shape, *dtype, name)));
    }
    auto [it, inserted] = tensors.emplace(
        name, TensorInfo{*dtype, std::move(shape), std::span<const std::byte>(data + begin, end - begin)});
    extents.push_back({begin, end, &it->first});
  }
  std::ranges::sort(extents, {}, &Extent::begin);
  std::uint64_t cursor = 0;
  for (const auto& extent : extents) {
    if (extent.begin != cursor) {
      throw std::runtime_error(std::format("tensor {}: starts at {} but the previous tensor ends at {}",
                                           *extent.name, extent.begin, cursor));
    }
    cursor = extent.end;
  }
  if (cursor != data_size) {
    throw std::runtime_error(std::format("tensors end at {} but the data section is {} bytes", cursor, data_size));
  }
  return tensors;
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(std::format("{}: cannot open: {}", path.string(), std::strerror(errno)));
  }
  struct stat status {};
  if (::fstat(fd, &status) != 0) {
    const int error = errno;
    ::close(fd);
    throw std::runtime_error(std::format("{}: cannot stat: {}", path.string(), std::strerror(error)));
  }
  const auto file_size = static_cast<std::size_t>(status.st_size);
  if (file_size < 8) {
    ::close(fd);
    throw std::runtime_error(std::format("{}: {} bytes is too short for a safetensors file", path.string(), file_size));
  }
  void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  const int map_error = errno;
  ::close(fd);
  if (mapped == MAP_FAILED) {
    throw std::runtime_error(std::format("{}: mmap failed: {}", path.string(), std::strerror(map_error)));
  }
  mapping_ = static_cast<const std::byte*>(mapped);
  mapping_size_ = file_size;
  try {
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, mapping_, sizeof(header_size));  // little-endian on every target
    if (header_size > kMaxHeaderBytes || header_size > file_size - 8) {
      throw std::runtime_error(std::format("header length {} does not fit the file", header_size));
    }
    const std::string_view header(reinterpret_cast<const char*>(mapping_ + 8), header_size);
    const std::byte* data = mapping_ + 8 + header_size;
    tensors_ = parse_safetensors_header(header, data, file_size - 8 - header_size);
  } catch (const std::exception& error) {
    release();
    throw std::runtime_error(std::format("{}: {}", path.string(), error.what()));
  }
}

SafetensorsFile::~SafetensorsFile() { release(); }

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : path_(std::move(other.path_)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      mapping_size_(std::exchange(other.mapping_size_, 0)),
      tensors_(std::move(other.tensors_)) {}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
  if (this != &other) {
    release();
    path_ = std::move(other.path_);
    mapping_ = std::exchange(other.mapping_, nullptr);
    mapping_size_ = std::exchange(other.mapping_size_, 0);
    tensors_ = std::move(other.tensors_);
  }
  return *this;
}

const TensorInfo* SafetensorsFile::find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

void SafetensorsFile::release() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(const_cast<std::byte*>(mapping_), mapping_size_);
    mapping_ = nullptr;
    mapping_size_ = 0;
  }
}

}  // namespace uocr
