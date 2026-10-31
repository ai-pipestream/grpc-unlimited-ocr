#pragma once

// Process configuration, entirely from GRPC_UOCR_* environment variables
// (design section 7). A malformed value throws at startup; nothing is
// silently defaulted away from what the operator wrote.

#include <cstddef>
#include <cstdint>
#include <string>

namespace uocr::service {

struct Config {
  std::string listen_address = "0.0.0.0:50073";
  // Compute backend: cuda, intel-gpu, metal, cpu or reference. Only the
  // ones this build carries are accepted.
  std::string backend = "cuda";
  std::uint32_t device_index = 0;
  // bf16, int8 or int4 decoder weights.
  std::string weight_format = "bf16";
  std::string models_dir = "/models";
  // Pages per decode window, and the per-page decode step cap, when a
  // request leaves them 0.
  std::uint32_t page_window = 10;
  std::uint32_t max_tokens_per_page = 20000;
  // The no-repeat n-gram rule: n, and the window of output it looks back
  // over for multi-page and single-page documents (upstream's defaults).
  std::uint32_t ngram_size = 35;
  std::uint32_t ngram_window = 1024;
  std::uint32_t ngram_window_single_page = 128;
  // Per-stream caps.
  std::uint32_t max_pages = 512;
  std::size_t max_page_bytes = 32ULL * 1024 * 1024;
  // Bytes of page rasters one stream may hold read but not yet decoded,
  // and the same across every stream. A stream at its cap stops reading,
  // so gRPC flow control holds the client back.
  std::size_t max_stream_buffered_bytes = 4 * (32ULL * 1024 * 1024);
  std::size_t max_buffered_bytes = 16 * (32ULL * 1024 * 1024);
  // Documents decoding at once; further streams wait for a slot.
  std::uint32_t documents_in_flight = 1;
};

// Throws std::invalid_argument naming the variable when a value is
// malformed or out of range.
Config load_config_from_env();

}  // namespace uocr::service
