// The Intel SYCL backend on the golden page, with bf16 weights,
// judged as a reduced-precision backend is: the same text, labels and
// structure as upstream fp32, every box coordinate within 2 of 1000. Token-exact parity is the reference backend's gate
// (golden_test). Needs UOCR_MODELS_DIR and a device; skips (77) otherwise.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#include "backend/intel/native/context.h"
#include "check.h"
#include "image/page_preprocess.h"
#include "intel_numeric_checks.h"
#include "model/intel/intel_decoder.h"
#include "model/intel/intel_vision.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"
#include "weights/sharded_weights.h"

using namespace uocr;

namespace {

std::string text_of(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  require(static_cast<bool>(in), "missing " + p.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Output with every box replaced by "[]", and the box coordinates in order.
std::pair<std::string, std::vector<int>> split_boxes(const std::string& s) {
  static const std::regex box(R"(\[(-?\d+), (-?\d+), (-?\d+), (-?\d+)\])");
  std::vector<int> coords;
  for (std::sregex_iterator it(s.begin(), s.end(), box), end; it != end; ++it) {
    for (int g = 1; g <= 4; ++g) coords.push_back(std::stoi((*it)[g].str()));
  }
  return {std::regex_replace(s, box, "[]"), coords};
}

class ObservedVision final : public VisionModel {
 public:
  ObservedVision(const intel_native::Context& ctx, const WeightSource& weights) : ctx_(ctx), vision_(ctx, weights) {
    newline_ = weights.get("model.image_newline");
    separator_ = weights.get("model.view_seperator");
  }
  Tensor encode_view(const Tensor& image) const override {
    const auto k = ctx_.kernel_submissions(), g = ctx_.gemm_submissions();
    const auto start = std::chrono::steady_clock::now();
    auto result = vision_.encode_view(image);
    ctx_.synchronize();
    for (float v : result.span()) require(std::isfinite(v), "vision output is finite");
    if (first_features.empty()) {
      const std::filesystem::path path = UOCR_GOLDENS "/quickstart-p2/projector.f32";
      const auto bytes = std::filesystem::file_size(path);
      require(bytes > 0 && bytes % sizeof(float) == 0, "reference vision feature file shape");
      std::vector<float> expected(bytes / sizeof(float));
      std::ifstream reference(path, std::ios::binary);
      reference.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(bytes));
      require(reference.good(), "complete reference vision features read");
      const auto error = intel_testing::finite_relative_error(result.span(), expected);
      std::println("VISION upstream_fp32_relative_error={:.9g}", error);
      // Upstream bf16 differs from its fp32 features by 0.057 on this page.
      require(error < 5.8e-2, "real vision matches upstream at its bf16 feature tolerance");
      first_features.assign(result.span().begin(), result.span().end());
    } else {
      const auto drift = intel_testing::finite_relative_error(result.span(), first_features);
      std::println("VISION repeated relative_error={:.9g}", drift);
      require(drift == 0, "vision features bitwise deterministic across requests");
    }
    std::println("VISION device={} native_kernels={} gpu_gemms={} seconds={:.6f}", ctx_.device_name(),
                 ctx_.kernel_submissions() - k, ctx_.gemm_submissions() - g,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    require(ctx_.kernel_submissions() > k && ctx_.gemm_submissions() > g,
            "vision executed native GPU kernels and GEMMs");
    vision_k = ctx_.kernel_submissions();
    vision_g = ctx_.gemm_submissions();
    return result;
  }
  mutable std::uint64_t vision_k = 0, vision_g = 0;
  mutable std::vector<float> first_features;

 private:
  const intel_native::Context& ctx_;
  intel_native::IntelVisionEncoder vision_;
};
}  // namespace

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) return skip("UOCR_MODELS_DIR not set");
  if (intel_native::device_count() == 0) return skip("no Intel SYCL device");
  return run_tests([&] {
    const std::filesystem::path golden = UOCR_GOLDENS "/quickstart-p2";
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const auto tok = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    const intel_native::Context ctx(0, 12ULL << 30);
    const ObservedVision vision(ctx, weights);
    const std::string png = text_of(golden / "page.png");
    const auto page = preprocess_page(decode_image({reinterpret_cast<const std::uint8_t*>(png.data()), png.size()}));
    const auto [want, want_boxes] = split_boxes(text_of(golden / "output_raw.txt"));
    intel_native::IntelDecoder decoder(ctx, weights);
    std::println("actual_device={} precision=bf16_weights_fp32_activations memory_cap_bytes={}", ctx.device_name(),
                 ctx.memory_cap());
    std::vector<std::int64_t> first_tokens;
    std::uint64_t retained = 0;
    for (int repeat = 0; repeat < 3; ++repeat) {
      GenerationConfig config;
      config.max_new_tokens = 600;
      const auto start = std::chrono::steady_clock::now();
      auto prefilled = start;
      const auto result = generate(vision, decoder, {page}, config, {}, [&](std::size_t) {
        prefilled = std::chrono::steady_clock::now();
        ctx.synchronize();
        std::println("PREFILL device={} native_kernels={} gpu_gemms={}", ctx.device_name(),
                     ctx.kernel_submissions() - vision.vision_k, ctx.gemm_submissions() - vision.vision_g);
        require(ctx.kernel_submissions() > vision.vision_k && ctx.gemm_submissions() > vision.vision_g,
                "decoder prefill executed native GPU work");
      });
      ctx.synchronize();
      std::println("DECODER completed native_kernels={} gpu_gemms={}", ctx.kernel_submissions() - vision.vision_k,
                   ctx.gemm_submissions() - vision.vision_g);
      for (const auto& t : result.trace)
        require(std::isfinite(t.logprob) && std::isfinite(t.model_logprob), "decoder log probabilities finite");
      const auto end = std::chrono::steady_clock::now();
      const auto total = std::chrono::duration<double>(end - start).count();
      const auto decode = std::chrono::duration<double>(end - prefilled).count();
      std::println(
          "repeat={} total_s={:.6f} vision_prefill_s={:.6f} decode_s={:.6f} tokens={} decode_tok_s={:.3f} "
          "retained_bytes={} peak_bytes={}",
          repeat, total, total - decode, decode, result.tokens.size(),
          static_cast<double>(result.tokens.size()) / decode, ctx.allocated_bytes(), ctx.peak_bytes());
      const auto raw = tok.decode(result.tokens);
      std::println("RAW {}", raw);
      require(result.finish == FinishReason::kEndOfSequence, "the page ends with EOS");
      const auto [ours, our_boxes] = split_boxes(raw);
      require(ours == want, "text, labels and structure match upstream fp32");
      require(our_boxes.size() == want_boxes.size(), "same number of boxes");
      int worst = 0;
      for (std::size_t i = 0; i < want_boxes.size(); ++i)
        worst = std::max(worst, std::abs(our_boxes[i] - want_boxes[i]));
      std::println("bf16: {} boxes, largest coordinate difference {}", want_boxes.size() / 4, worst);
      require(worst <= 2, "boxes within 2 of 1000");
      if (repeat == 0) {
        first_tokens = result.tokens;
        retained = ctx.allocated_bytes();
      } else {
        require(first_tokens == result.tokens, "repeated inference is token deterministic");
        require(retained == ctx.allocated_bytes(), "native retained allocations plateau after warmup");
      }
      std::ifstream status("/proc/self/status");
      std::string line;
      while (std::getline(status, line))
        if (line.starts_with("VmRSS:") || line.starts_with("VmHWM:")) std::println("{}", line);
    }
  });
}
