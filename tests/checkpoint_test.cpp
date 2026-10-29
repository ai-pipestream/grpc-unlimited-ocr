// The published checkpoint matches the architecture table exactly. Needs
// the weights: set UOCR_MODELS_DIR to the directory holding
// model.safetensors.index.json; skips (77) without it.

#include "check.h"
#include "weights/architecture.h"
#include "weights/sharded_weights.h"

using namespace uocr;

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) {
    return skip("UOCR_MODELS_DIR not set");
  }
  return run_tests([&] {
    const auto weights = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const auto problems = check_architecture(unlimited_ocr2_tensors(), weights);
    for (const auto& problem : problems) {
      std::println(stderr, "{}", problem);
    }
    require(problems.empty(), std::format("{} architecture problems", problems.size()));
  });
}
