// Diagnostic oracle: native CPU reference decoder with committed upstream
// fp32 projector features for two copies of the known page. No GPU inference
// or new vision golden is claimed by this executable.
#include <cmath>
#include <fstream>

#include "backend/reference/parallel.h"
#include "check.h"
#include "model/decoder.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"
#include "weights/sharded_weights.h"

using namespace uocr;
namespace {
class RecordedVision final : public VisionModel {
 public:
  explicit RecordedVision(const WeightSource& weights) : projected_({256, 1280}) {
    newline_ = weights.get("model.image_newline");
    separator_ = weights.get("model.view_seperator");
    std::ifstream in(UOCR_GOLDENS "/quickstart-p2/projector.f32", std::ios::binary);
    in.read(reinterpret_cast<char*>(projected_.data()), projected_.numel() * sizeof(float));
    require(in.good() && in.peek() == std::char_traits<char>::eof(), "complete recorded fp32 projector shape");
    for (const auto value : projected_.span()) require(std::isfinite(value), "recorded feature finite");
  }
  Tensor encode_view(const Tensor&) const override { return projected_; }

 private:
  Tensor projected_;
};
}  // namespace
int main() {
  const auto* models = env_or_null("UOCR_MODELS_DIR");
  if (!models) return skip("model path absent");
  return run_tests([&] {
    reference::set_thread_count(2);
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const RecordedVision vision(weights);
    const Decoder model(weights);
    ReferenceDecoder decoder(model);
    const auto tokenizer = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    GenerationConfig config;
    config.max_page_tokens = 600;
    config.max_new_tokens = 2 * (config.max_page_tokens + 1) + 1;
    config.ngram_size = 35;
    config.ngram_window = 1024;
    PageViews page;
    page.global = Tensor({3, 1024, 1024});
    std::println("reference decoder, upstream fp32 recorded vision, 2 pages, 2 CPU threads");
    const auto result = generate(
        vision, decoder, {page, page}, config,
        [&](const GeneratedToken& step) {
          require(std::isfinite(step.logprob) && std::isfinite(step.model_logprob), "finite oracle logits");
          std::print("{}", tokenizer.decode({step.token}));
          std::fflush(stdout);
          return true;
        },
        [](std::size_t count) { std::println("reference prefill complete prompt_tokens={}", count); });
    require(result.finish == FinishReason::kEndOfSequence, "reference completed without token cap");
    std::println("\nREFERENCE_RAW {}", tokenizer.decode(result.tokens));
    std::println("reference complete tokens={}", result.tokens.size());
  });
}
