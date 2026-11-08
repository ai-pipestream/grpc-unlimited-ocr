// The CUDA backend on the golden page, with bf16 and with int8 weights,
// judged as a reduced-precision backend is: the same text, labels and
// structure as upstream fp32, every box coordinate within 2 of 1000. Token-exact parity is the reference backend's gate
// (golden_test). Needs UOCR_MODELS_DIR and a device; skips (77) otherwise.

#include <fstream>
#include <chrono>
#include <cstdio>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#include "backend/cuda/context.h"
#include "check.h"
#include "image/page_preprocess.h"
#include "model/cuda/cuda_decoder.h"
#include "model/cuda/cuda_vision.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"
#include "weights/sharded_weights.h"

using namespace uocr;

namespace {

void memory_phase(const char* phase) {
  const auto device = cuda::device_info(0);
  std::println(stderr, "phase={} device={} free_bytes={} total_bytes={} pool_reserved={} pool_used={}",
               phase, device.name, device.free_bytes, device.total_bytes,
               device.pool_reserved_bytes, device.pool_used_bytes);
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);)
    if (line.starts_with("VmRSS:") || line.starts_with("VmHWM:") ||
        line.starts_with("RssAnon:") || line.starts_with("RssFile:")) std::println(stderr, "{}", line);
  for (const auto* name : {"memory.current", "memory.peak", "memory.events"}) {
    std::ifstream in(std::string("/sys/fs/cgroup/") + name);
    for (std::string line; std::getline(in, line);) std::println(stderr, "{} {}", name, line);
  }
  std::fflush(stderr);
}

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
  if (cuda::device_count() == 0) return skip("no CUDA device");
  return run_tests([&] {
    const std::filesystem::path golden = UOCR_GOLDENS "/quickstart-p2";
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const auto tok = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    const cuda::Context ctx(0);
    memory_phase("before-vision-load");
    const cuda::CudaVisionEncoder vision(ctx, weights);
    memory_phase("after-vision-load");
    const std::string png = text_of(golden / "page.png");
    const auto page = preprocess_page(decode_image({reinterpret_cast<const std::uint8_t*>(png.data()), png.size()}));
    const auto [want, want_boxes] = split_boxes(text_of(golden / "output_raw.txt"));
    // bf16 needs about 8 GB on the device; an 8 GB Jetson shares that with
    // the OS, so it runs the int8 leg only.
    std::vector<cuda::WeightFormat> formats{cuda::WeightFormat::kInt8};
    if (cuda::device_info(0).total_bytes >= (12ULL << 30)) formats.insert(formats.begin(), cuda::WeightFormat::kBf16);
    for (const auto format : formats) {
      memory_phase(format == cuda::WeightFormat::kBf16 ? "before-bf16-decoder-load" : "before-int8-decoder-load");
      cuda::CudaDecoder decoder(ctx, weights, DecoderDims{}, cuda::CudaDecoder::default_router_params(), format);
      memory_phase("after-decoder-load");
      GenerationConfig config;
      config.max_new_tokens = 600;
      const auto start = std::chrono::steady_clock::now();
      const auto result = generate(vision, decoder, {page}, config);
      memory_phase("after-generation");
      std::println(stderr, "generation seconds={:.6f}",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      require(result.finish == FinishReason::kEndOfSequence, "the page ends with EOS");
      const auto [ours, our_boxes] = split_boxes(tok.decode(result.tokens));
      require(ours == want, "text, labels and structure match upstream fp32");
      require(our_boxes.size() == want_boxes.size(), "same number of boxes");
      int worst = 0;
      for (std::size_t i = 0; i < want_boxes.size(); ++i) worst = std::max(worst, std::abs(our_boxes[i] - want_boxes[i]));
      std::println("{}: {} boxes, largest coordinate difference {}",
                   format == cuda::WeightFormat::kBf16 ? "bf16" : "int8", want_boxes.size() / 4, worst);
      require(worst <= 2, "boxes within 2 of 1000");
    }
  });
}
