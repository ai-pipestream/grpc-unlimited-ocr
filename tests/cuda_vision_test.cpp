// The CUDA vision encoder against the reference one: on a tiny random
// model (base and tiled layouts), and with the real weights on the golden
// page (when UOCR_MODELS_DIR is set). bf16 weights and GEMM inputs bound
// the agreement, so tolerances are bf16-sized. Skips (77) without a device.

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>

#include "backend/cuda/context.h"
#include "check.h"
#include "image/page_preprocess.h"
#include "model/cuda/cuda_vision.h"
#include "model/vision_encoder.h"
#include "tensor/dtype.h"
#include "weights/architecture.h"
#include "weights/sharded_weights.h"

using namespace uocr;

namespace {

double relative_error(std::span<const float> got, std::span<const float> want) {
  double worst = 0.0, scale = 1e-12;
  for (std::size_t i = 0; i < want.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
  }
  return worst / scale;
}

ModelDims tiny_dims() {
  ModelDims d;
  d.sam.width = 24;
  d.sam.layers = 2;
  d.sam.heads = 2;
  d.sam.mlp = 48;
  d.sam.global_layers = {1, 99, 99, 99};
  d.sam.neck = 8;
  d.sam.net2 = 12;
  d.sam.net3 = 16;
  d.clip.width = 16;
  d.clip.layers = 2;
  d.clip.heads = 2;
  d.clip.mlp = 32;
  d.projector_in = 32;
  d.decoder.hidden = 20;
  return d;
}

// Random parameters, already on the bf16 grid so both backends see the same values.
TableWeights tiny_weights(const ModelDims& dims) {
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 0.05f);
  TableWeights table;
  for (const auto& spec : unlimited_ocr2_tensors(dims)) {
    if (!spec.name.starts_with("model.sam_model") && !spec.name.starts_with("model.vision_model") &&
        !spec.name.starts_with("model.projector") && spec.name != "model.image_newline" &&
        spec.name != "model.view_seperator") {
      continue;
    }
    Tensor t(spec.shape);
    const bool norm = spec.name.find("norm") != std::string::npos && spec.name.ends_with(".weight");
    for (float& v : t.span()) v = bf16_to_f32(f32_to_bf16(norm ? 1.0f + normal(rng) : normal(rng)));
    table.put(spec.name, std::move(t));
  }
  return table;
}

Tensor random_image(std::int64_t size, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  Tensor img({3, size, size});
  for (float& v : img.span()) v = bf16_to_f32(f32_to_bf16(unit(rng)));
  return img;
}

std::vector<float> floats_of(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<float> out(raw.size() / 4);
  std::memcpy(out.data(), raw.data(), out.size() * 4);
  return out;
}

}  // namespace

int main() {
  if (cuda::device_count() == 0) return skip("no CUDA device");
  return run_tests([] {
    const cuda::Context ctx(0);
    const auto dims = tiny_dims();
    const auto weights = tiny_weights(dims);
    const VisionEncoder ref(weights, dims);
    const cuda::CudaVisionEncoder dev(ctx, weights, dims);
    PageViews tiled{random_image(128, 3), {random_image(128, 4), random_image(128, 5)}, 2, 1};
    const Tensor want = ref.encode_page(tiled), got = dev.encode_page(tiled);
    require(got.shape() == want.shape(), "same token layout");
    const double tiny_err = relative_error(got.span(), want.span());
    std::println("tiny model, tiled page: relative error {:.2e}", tiny_err);
    require(tiny_err < 2e-2, "tiny model agrees at bf16 tolerance");

    const char* models = env_or_null("UOCR_MODELS_DIR");
    if (models == nullptr) return;
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights real(shards);
    const cuda::CudaVisionEncoder vision(ctx, real);
    std::ifstream in(UOCR_GOLDENS "/quickstart-p2/page.png", std::ios::binary);
    const std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const PageViews page = preprocess_page(decode_image(png));
    const auto start = std::chrono::steady_clock::now();
    const Tensor features = vision.encode_view(page.global);
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double err = relative_error(features.span(), floats_of(UOCR_GOLDENS "/quickstart-p2/projector.f32"));
    std::println("real weights, golden page: relative error {:.2e} vs fp32 upstream, {:.0f} ms (first call)", err, ms);
    // Upstream's own bf16 run lands 5.7e-2 (max) from its fp32 run on this
    // page; bf16 weights cannot do better than that order, so the bar is it.
    require(err < 5.8e-2, "real vision features drift no more than upstream's own bf16 run");
  });
}
