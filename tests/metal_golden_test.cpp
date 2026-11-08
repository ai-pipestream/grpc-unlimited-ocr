// The Metal backend on the golden page, with upstream bf16 weights,
// judged as a reduced-precision backend is: the same text, labels and
// structure as upstream fp32, every box coordinate within 2 of 1000. Token-exact parity is the reference backend's gate
// (golden_test). Needs UOCR_MODELS_DIR and a device; skips (77) otherwise.

#include <fstream>
#include <chrono>
#include <sys/resource.h>
#include "backend/reference/parallel.h"
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#include "backend/metal/context.h"
#include "check.h"
#include "image/page_preprocess.h"
#include "model/metal/metal_decoder.h"
#include "model/metal/metal_vision.h"
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

}  // namespace

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) return skip("UOCR_MODELS_DIR not set");
  if (metal::device_count() == 0) return skip("no Metal device");
  return run_tests([&] {
    reference::set_thread_count(2);
    const std::filesystem::path golden = UOCR_GOLDENS "/quickstart-p2";
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const auto tok = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    const metal::Context ctx(0);
    const metal::MetalVisionEncoder vision(ctx, weights);
    std::println("device={} weights=upstream-bf16 activations=fp32 accumulation=fp32", metal::device_info(0).name);
    std::fflush(stdout);
    const std::string png = text_of(golden / "page.png");
    const auto page = preprocess_page(decode_image({reinterpret_cast<const std::uint8_t*>(png.data()), png.size()}));
    const auto [want, want_boxes] = split_boxes(text_of(golden / "output_raw.txt"));
    for (const auto format : {metal::WeightFormat::kBf16}) {
      metal::MetalDecoder decoder(ctx, weights, DecoderDims{}, metal::MetalDecoder::default_router_params(), format);
      GenerationConfig config;
      config.max_new_tokens = 600;
      std::vector<std::int64_t> previous;
      std::size_t retained = 0, device_retained = 0;
      for (int repeat = 0; repeat < 2; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      auto prefilled = start;
      const auto kernels_before = ctx.kernel_dispatches();
      const auto gemms_before = ctx.gemm_dispatches();
      std::size_t steps = 0;
      const auto result = generate(vision, decoder, {page}, config, [&](const GeneratedToken&) {
        if (++steps % 32 == 0) { std::println("repeat={} steps={}", repeat, steps); std::fflush(stdout); }
        return true;
      }, [&](std::size_t n) {
        prefilled = std::chrono::steady_clock::now();
        std::println("repeat={} prefill_tokens={} prefill_seconds={:.3f}", repeat, n,
                     std::chrono::duration<double>(prefilled-start).count());
        std::fflush(stdout);
      });
      const auto end = std::chrono::steady_clock::now();
      rusage usage{};
      require(getrusage(RUSAGE_SELF, &usage) == 0, "read process memory");
      std::println("repeat={} seconds={:.3f} decode_seconds={:.3f} tokens={} peak_rss_bytes={} metal_live_bytes={} metal_peak_bytes={} metal_device_allocated_bytes={} kernels={} gemms={}",
        repeat, std::chrono::duration<double>(end-start).count(), std::chrono::duration<double>(end-prefilled).count(),
        result.tokens.size(), usage.ru_maxrss, ctx.allocated_bytes(), ctx.peak_bytes(), ctx.device_allocated_bytes(),
        ctx.kernel_dispatches()-kernels_before, ctx.gemm_dispatches()-gemms_before);
      require(ctx.kernel_dispatches() > kernels_before && ctx.gemm_dispatches() > gemms_before, "native Metal commands executed");
      if (repeat) require(result.tokens == previous, "repeated page token determinism");
      previous = result.tokens;
      if (repeat) {
        require(ctx.allocated_bytes() == retained, "owned buffers return to the same retained size");
        require(ctx.device_allocated_bytes() <= device_retained + (64ULL << 20), "device allocations remain bounded after warmup");
      }
      retained = ctx.allocated_bytes();
      device_retained = ctx.device_allocated_bytes();


      require(result.finish == FinishReason::kEndOfSequence, "the page ends with EOS");
      const auto [ours, our_boxes] = split_boxes(tok.decode(result.tokens));
      require(ours == want, "text, labels and structure match upstream fp32");
      require(our_boxes.size() == want_boxes.size(), "same number of boxes");
      int worst = 0;
      for (std::size_t i = 0; i < want_boxes.size(); ++i) worst = std::max(worst, std::abs(our_boxes[i] - want_boxes[i]));
      std::println("{}: {} boxes, largest coordinate difference {}",
                   "bf16", want_boxes.size() / 4, worst);
      require(worst <= 2, "boxes within 2 of 1000");
      std::fflush(stdout);
      }
    }
  });
}
