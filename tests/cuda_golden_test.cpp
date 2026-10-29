// The CUDA backend on the golden page, judged as a bf16 backend is: the
// same text, labels and structure as upstream fp32, every box coordinate
// within 2 of 1000. Token-exact parity is the reference backend's gate
// (golden_test). Needs UOCR_MODELS_DIR and a device; skips (77) otherwise.

#include <fstream>
#include <iterator>
#include <regex>
#include <string>

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
    const cuda::CudaVisionEncoder vision(ctx, weights);
    cuda::CudaDecoder decoder(ctx, weights);
    const std::string png = text_of(golden / "page.png");
    const auto page = preprocess_page(decode_image({reinterpret_cast<const std::uint8_t*>(png.data()), png.size()}));
    GenerationConfig config;
    config.max_new_tokens = 600;
    const auto result = generate(vision, decoder, {page}, config);
    require(result.finish == FinishReason::kEndOfSequence, "the page ends with EOS");

    const auto [ours, our_boxes] = split_boxes(tok.decode(result.tokens));
    const auto [want, want_boxes] = split_boxes(text_of(golden / "output_raw.txt"));
    require(ours == want, "text, labels and structure match upstream fp32");
    require(our_boxes.size() == want_boxes.size(), "same number of boxes");
    int worst = 0;
    for (std::size_t i = 0; i < want_boxes.size(); ++i) worst = std::max(worst, std::abs(our_boxes[i] - want_boxes[i]));
    std::println("{} boxes, largest coordinate difference {}", want_boxes.size() / 4, worst);
    require(worst <= 2, "boxes within 2 of 1000");
  });
}
