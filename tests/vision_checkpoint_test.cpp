// The real vision encoder on one synthetic 1024 page: shape and finiteness
// only (parity with upstream is the goldens' job). Needs UOCR_MODELS_DIR;
// skips (77) without it.

#include <chrono>
#include <cmath>
#include <random>

#include "check.h"
#include "model/vision_encoder.h"
#include "weights/sharded_weights.h"

using namespace uocr;

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) {
    return skip("UOCR_MODELS_DIR not set");
  }
  return run_tests([&] {
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const VisionEncoder vision(weights);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    PageViews page{Tensor({3, 1024, 1024}), {}, 1, 1};
    for (float& v : page.global.span()) v = unit(rng);
    const auto start = std::chrono::steady_clock::now();
    const Tensor tokens = vision.encode_page(page);
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::println("base page: {} x {} in {:.1f} s", tokens.dim(0), tokens.dim(1), seconds);
    require(tokens.dim(0) == 273 && tokens.dim(1) == 1280, "a base page is 273 tokens of 1280");
    for (const float v : tokens.span()) require(std::isfinite(v), "finite tokens");
  });
}
