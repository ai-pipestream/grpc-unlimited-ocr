#include "service/model_engine.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <stb_image.h>

#include "image/page_preprocess.h"
#include "model/decoder.h"
#include "model/vision_encoder.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"
#include "weights/architecture.h"
#include "weights/sharded_weights.h"

#if UOCR_WITH_CUDA
#include "backend/cuda/context.h"
#include "model/cuda/cuda_decoder.h"
#include "model/cuda/cuda_vision.h"
#endif

#ifndef UOCR_MANIFEST_SHA256
#define UOCR_MANIFEST_SHA256 ""
#endif

namespace uocr::service {
namespace {

StepRule step_rule(Intervention i) {
  switch (i) {
    case Intervention::kNone: return StepRule::kNone;
    case Intervention::kRepeatBan: return StepRule::kRepeatBan;
    case Intervention::kPageGuard: return StepRule::kPageGuard;
  }
  return StepRule::kNone;
}

WindowFinish window_finish(FinishReason f) {
  switch (f) {
    case FinishReason::kEndOfSequence: return WindowFinish::kEndOfSequence;
    case FinishReason::kTokenBudget: return WindowFinish::kTokenBudget;
    case FinishReason::kContextLength: return WindowFinish::kContextLength;
  }
  return WindowFinish::kTokenBudget;
}

class ModelEngine final : public Engine {
 public:
  explicit ModelEngine(const Config& config)
      : config_(config),
        shards_(ShardedWeights::open(std::filesystem::path(config.models_dir) / "model.safetensors.index.json")),
        tokenizer_(Tokenizer::load(std::filesystem::path(config.models_dir) / "tokenizer.json")) {
    const auto problems = check_architecture(unlimited_ocr2_tensors(), shards_);
    if (!problems.empty()) throw std::runtime_error("the weights do not match the architecture: " + problems.front());
    weights_ = std::make_unique<CheckpointWeights>(shards_);
    page_token_ = static_cast<std::uint32_t>(tokenizer_.special_id("<PAGE>"));
    build_backend();
  }

  EngineInfo info() const override {
    return {config_.backend, device_, config_.weight_format, UOCR_MANIFEST_SHA256};
  }

  std::string piece(std::uint32_t token) const override { return tokenizer_.piece(token); }
  std::string bytes(std::uint32_t token) const override { return tokenizer_.decode({token}); }
  std::uint32_t page_token() const override { return page_token_; }
  std::uint32_t eos_token() const override { return 1; }

  PageSize probe(std::string_view raster) const override {
    int w = 0;
    int h = 0;
    int channels = 0;
    const auto* data = reinterpret_cast<const stbi_uc*>(raster.data());
    if (raster.size() > static_cast<std::size_t>(INT32_MAX) ||
        stbi_info_from_memory(data, static_cast<int>(raster.size()), &w, &h, &channels) == 0) {
      throw ImageError(std::string("not a readable PNG or JPEG: ") + stbi_failure_reason());
    }
    return {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
  }

  WindowResult decode(const WindowRequest& request, const std::function<void(std::uint64_t)>& started,
                      const std::function<bool(const EngineStep&)>& step) override {
    PreprocessOptions pre;
    pre.tiled = request.tiled;
    std::vector<PageViews> views;
    for (const std::string_view raster : request.pages) {
      const auto* bytes = reinterpret_cast<const std::uint8_t*>(raster.data());
      views.push_back(preprocess_page(decode_image({bytes, raster.size()}), pre));
    }
    const auto pages = static_cast<std::int64_t>(request.pages.size());
    GenerationConfig gen;
    gen.max_page_tokens = request.max_tokens_per_page;
    gen.max_new_tokens = pages * (gen.max_page_tokens + 1) + 1;
    gen.ngram_size = config_.ngram_size;
    gen.ngram_window = pages > 1 ? config_.ngram_window : config_.ngram_window_single_page;
    bool stopped = false;
    std::uint64_t prompt = 0;
    const GenerationResult result = generate(
        *vision_, *decoder_, views, gen,
        [&](const GeneratedToken& t) {
          const EngineStep s{static_cast<std::uint32_t>(t.token), t.logprob, static_cast<std::uint32_t>(t.model_token),
                             t.model_logprob, step_rule(t.intervention)};
          stopped = !step(s);
          return !stopped;
        },
        [&](std::size_t n) {
          prompt = n;
          started(n);
        });
    return {prompt, stopped && result.finish != FinishReason::kEndOfSequence ? WindowFinish::kStopped
                                                                             : window_finish(result.finish)};
  }

 private:
  void build_backend() {
    if (config_.backend == "reference") {
      vision_ = std::make_unique<VisionEncoder>(*weights_);
      reference_decoder_ = std::make_unique<Decoder>(*weights_);
      decoder_ = std::make_unique<ReferenceDecoder>(*reference_decoder_);
      device_ = "cpu (fp32 reference)";
      return;
    }
#if UOCR_WITH_CUDA
    if (config_.backend == "cuda") {
      if (config_.weight_format != "bf16" && config_.weight_format != "int8") {
        throw std::runtime_error("the CUDA backend takes bf16 or int8 weights, not " + config_.weight_format);
      }
      cuda_ = std::make_unique<cuda::Context>(static_cast<int>(config_.device_index));
      vision_ = std::make_unique<cuda::CudaVisionEncoder>(*cuda_, *weights_);
      decoder_ = std::make_unique<cuda::CudaDecoder>(
          *cuda_, *weights_, DecoderDims{}, cuda::CudaDecoder::default_router_params(),
          config_.weight_format == "int8" ? cuda::WeightFormat::kInt8 : cuda::WeightFormat::kBf16);
      device_ = "cuda:" + std::to_string(config_.device_index);
      return;
    }
#endif
    throw std::runtime_error("backend " + config_.backend + " is not built into this binary");
  }

  const Config& config_;
  ShardedWeights shards_;
  Tokenizer tokenizer_;
  std::unique_ptr<CheckpointWeights> weights_;
#if UOCR_WITH_CUDA
  std::unique_ptr<cuda::Context> cuda_;  // declared before the models: device arrays go first
#endif
  std::unique_ptr<VisionModel> vision_;
  std::unique_ptr<Decoder> reference_decoder_;
  std::unique_ptr<DecoderModel> decoder_;
  std::uint32_t page_token_ = 128827;
  std::string device_;
};

}  // namespace

std::unique_ptr<Engine> load_model_engine(const Config& config) { return std::make_unique<ModelEngine>(config); }

}  // namespace uocr::service
