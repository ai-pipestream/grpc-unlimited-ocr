#pragma once

// What the service needs from the model, and nothing more: the vocabulary
// to turn token ids into text, a way to read a page raster's size, and a
// decode of one page window that reports every step as it is chosen. The
// real engine (model_engine.h) runs the native forward pass; service tests
// put a scripted fake behind the same interface, with no weights and no
// device.

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace uocr::service {

// Facts GetServiceInfo reports.
struct EngineInfo {
  std::string backend;  // "cuda", "reference", ...
  std::string device;
  std::string weight_format;
  std::string manifest_sha256;
};

struct PageSize {
  std::uint32_t width_px = 0;
  std::uint32_t height_px = 0;
};

// A page raster the engine cannot read.
class ImageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class StepRule { kNone, kRepeatBan, kPageGuard };

// One decode step, exactly as the model and the logit rules produced it.
struct EngineStep {
  std::uint32_t token = 0;
  float logprob = 0.0f;
  std::uint32_t model_token = 0;
  float model_logprob = 0.0f;
  StepRule rule = StepRule::kNone;
};

enum class WindowFinish { kEndOfSequence, kTokenBudget, kContextLength, kStopped };

struct WindowRequest {
  std::vector<std::string_view> pages;  // rasters, in page order
  bool tiled = false;
  std::uint32_t max_tokens_per_page = 0;
};

struct WindowResult {
  std::uint64_t prompt_tokens = 0;
  WindowFinish finish = WindowFinish::kEndOfSequence;
};

class Engine {
 public:
  virtual ~Engine() = default;

  virtual EngineInfo info() const = 0;

  // The vocabulary entry of `token`, before byte decoding ("Ġpage").
  virtual std::string piece(std::uint32_t token) const = 0;
  // The bytes `token` decodes to; special tokens give their literal text.
  virtual std::string bytes(std::uint32_t token) const = 0;
  virtual std::uint32_t page_token() const = 0;
  virtual std::uint32_t eos_token() const = 0;

  // The raster's size. Throws ImageError when it is not a readable PNG or
  // JPEG.
  virtual PageSize probe(std::string_view raster) const = 0;

  // Decodes one window: prefill, then one call of `step` per decode step,
  // the closing EOS step included. `started` is called once, after
  // prefill, with the prompt length. Returning false from `step` stops the
  // decode (finish kStopped). Throws on an engine failure. One window at a
  // time per engine; the service serialises calls.
  virtual WindowResult decode(const WindowRequest& request, const std::function<void(std::uint64_t)>& started,
                              const std::function<bool(const EngineStep&)>& step) = 0;
};

}  // namespace uocr::service
