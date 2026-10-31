#include "service/config.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>

namespace uocr::service {
namespace {

std::uint64_t configured_number(const char* name, std::uint64_t fallback, std::uint64_t minimum,
                                std::uint64_t maximum) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || *end != '\0' || raw[0] == '-' || value < minimum || value > maximum) {
    throw std::invalid_argument(std::string(name) + " must be an integer between " + std::to_string(minimum) +
                                " and " + std::to_string(maximum));
  }
  return value;
}

std::uint32_t configured_u32(const char* name, std::uint32_t fallback, std::uint32_t minimum, std::uint32_t maximum) {
  return static_cast<std::uint32_t>(configured_number(name, fallback, minimum, maximum));
}

std::string configured_choice(const char* name, const std::string& fallback,
                              std::initializer_list<std::string_view> choices) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  const std::string value(raw);
  if (std::ranges::find(choices, value) == choices.end()) {
    std::string list;
    for (const auto c : choices) list += (list.empty() ? "" : ", ") + std::string(c);
    throw std::invalid_argument(std::string(name) + " must be one of " + list);
  }
  return value;
}

std::string configured_string(const char* name, const std::string& fallback) {
  const char* raw = std::getenv(name);
  return raw == nullptr || *raw == '\0' ? fallback : raw;
}

}  // namespace

Config load_config_from_env() {
  constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
  Config c;
  c.listen_address = configured_string("GRPC_UOCR_LISTEN_ADDRESS", c.listen_address);
  c.backend = configured_choice("GRPC_UOCR_BACKEND", c.backend, {"cuda", "intel-gpu", "metal", "cpu", "reference"});
  c.device_index = configured_u32("GRPC_UOCR_DEVICE_INDEX", c.device_index, 0, 63);
  c.weight_format = configured_choice("GRPC_UOCR_WEIGHT_FORMAT", c.weight_format, {"bf16", "int8", "int4"});
  c.models_dir = configured_string("GRPC_UOCR_MODELS_DIR", c.models_dir);
  c.page_window = configured_u32("GRPC_UOCR_PAGE_WINDOW", c.page_window, 1, 64);
  c.max_tokens_per_page = configured_u32("GRPC_UOCR_MAX_TOKENS_PER_PAGE", c.max_tokens_per_page, 16, 131072);
  c.ngram_size = configured_u32("GRPC_UOCR_NGRAM_SIZE", c.ngram_size, 0, 1024);
  c.ngram_window = configured_u32("GRPC_UOCR_NGRAM_WINDOW", c.ngram_window, 1, 131072);
  c.ngram_window_single_page =
      configured_u32("GRPC_UOCR_NGRAM_WINDOW_SINGLE_PAGE", c.ngram_window_single_page, 1, 131072);
  c.max_pages = configured_u32("GRPC_UOCR_MAX_PAGES", c.max_pages, 1, 100000);
  c.max_page_bytes = configured_number("GRPC_UOCR_MAX_PAGE_BYTES", c.max_page_bytes, 1024, kGiB);
  c.max_stream_buffered_bytes = configured_number("GRPC_UOCR_MAX_STREAM_BUFFERED_BYTES", 4 * c.max_page_bytes,
                                                  c.max_page_bytes, 64 * kGiB);
  c.max_buffered_bytes = configured_number("GRPC_UOCR_MAX_BUFFERED_BYTES", 16 * c.max_page_bytes,
                                           c.max_stream_buffered_bytes, 256 * kGiB);
  c.documents_in_flight = configured_u32("GRPC_UOCR_DOCUMENTS_IN_FLIGHT", c.documents_in_flight, 1, 64);
  return c;
}

}  // namespace uocr::service
