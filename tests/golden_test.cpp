// Parity with the upstream reference implementation, on pages recorded
// once from it (tests/goldens/<name>/, see tests/goldens/README.md):
// preprocessing pixel for pixel, the vision features within tolerance,
// the prompt ids exactly, and the generated tokens exactly. Needs
// UOCR_MODELS_DIR; skips (77) without it.

#include <cmath>
#include <cstring>
#include <span>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "image/page_preprocess.h"
#include "model/decoder.h"
#include "model/vision_encoder.h"
#include "runtime/generator.h"
#include "weights/sharded_weights.h"

using namespace uocr;

namespace {

std::vector<std::uint8_t> bytes_of(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  require(static_cast<bool>(in), "missing golden file " + p.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<std::int64_t> ids_of(const std::filesystem::path& p) {
  std::ifstream in(p);
  require(static_cast<bool>(in), "missing golden file " + p.string());
  std::vector<std::int64_t> ids;
  for (std::int64_t v = 0; in >> v;) ids.push_back(v);
  return ids;
}

std::vector<float> floats_of(const std::filesystem::path& p) {
  const auto raw = bytes_of(p);
  std::vector<float> out(raw.size() / 4);
  std::memcpy(out.data(), raw.data(), out.size() * 4);
  return out;
}

// Largest absolute difference relative to the golden's largest magnitude.
double relative_error(std::span<const float> ours, std::span<const float> golden) {
  require(ours.size() == golden.size(), "feature sizes differ");
  double worst = 0.0, scale = 1e-12;
  for (std::size_t i = 0; i < ours.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(ours[i]) - golden[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(golden[i])));
  }
  return worst / scale;
}

}  // namespace

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) return skip("UOCR_MODELS_DIR not set");
  const char* golden_env = env_or_null("UOCR_GOLDEN_DIR");
  const std::filesystem::path golden = golden_env != nullptr ? golden_env : UOCR_GOLDENS "/quickstart-p2";
  return run_tests([&] {
    const RgbImage page = decode_image(bytes_of(golden / "page.png"));
    const RgbImage expected = decode_image(bytes_of(golden / "letterbox.png"));
    const RgbImage ours = letterbox(page, 1024, 127);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < ours.pixels.size(); ++i) differing += ours.pixels[i] != expected.pixels[i];
    std::println("letterbox: {} of {} channel values differ", differing, ours.pixels.size());
    require(differing == 0, "letterbox matches the reference pixel for pixel");

    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const VisionEncoder vision(weights);
    const PageViews views = preprocess_page(page);
    const Tensor projected = vision.encode_view(views.global);
    const double err = relative_error(projected.span(), floats_of(golden / "projector.f32"));
    std::println("projector features: relative error {:.3e}", err);
    require(err < 1e-3, "vision features match the reference");

    const Decoder reference_decoder(weights);
    ReferenceDecoder decoder(reference_decoder);
    GenerationConfig config;
    const auto golden_tokens = ids_of(golden / "tokens.txt");
    config.max_new_tokens = static_cast<std::int64_t>(golden_tokens.size()) + 1;
    const auto result = generate(vision, decoder, {views}, config);
    require(result.prompt == ids_of(golden / "prompt_ids.txt"), "prompt ids match the reference");
    std::size_t same = 0;
    while (same < std::min(result.tokens.size(), golden_tokens.size()) && result.tokens[same] == golden_tokens[same]) ++same;
    std::println("tokens: {} of {} match before the first difference", same, golden_tokens.size());
    require(result.tokens == golden_tokens, "generated tokens match the reference exactly");
  });
}
