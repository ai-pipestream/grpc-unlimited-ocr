// Diagnostic only, not a qualification gate. Existing vision/golden thresholds
// remain in their tests. Preserve actual intermediate arrays for comparisons.
#include <cmath>
#include <fstream>
#include <iterator>

#include "check.h"
#include "image/page_preprocess.h"
#include "intel_numeric_checks.h"
#include "model/intel/intel_vision.h"
#include "weights/sharded_weights.h"

namespace uocr::intel_native {
struct IntelVisionTestAccess {
  static Tensor sam(const IntelVisionEncoder& model, const Tensor& image) {
    const Array input = upload_tensor(model.ctx_, image, DType::kF32);
    std::int64_t grid = 0;
    const auto output = model.sam_forward(input, image.dim(1), grid);
    return Tensor({grid * grid, model.dims_.sam.net3}, output.download());
  }
};
}  // namespace uocr::intel_native
using namespace uocr;
namespace {
std::vector<float> floats(const std::filesystem::path& path) {
  const auto bytes = std::filesystem::file_size(path);
  require(bytes > 0 && bytes % sizeof(float) == 0, "reference shape");
  std::vector<float> result(bytes / sizeof(float));
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  require(in.good(), "reference read");
  return result;
}
void report(const char* stage, const Tensor& actual, const std::vector<float>& expected, const std::string& prefix) {
  const auto error = intel_testing::finite_relative_error(actual.span(), expected);
  double delta = 0, scale = 0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double diff = static_cast<double>(actual.data()[i]) - expected[i];
    delta += diff * diff;
    scale += static_cast<double>(expected[i]) * expected[i];
    if (std::abs(diff) > std::abs(static_cast<double>(actual.data()[worst]) - expected[worst])) worst = i;
  }
  std::println("DIAGNOSTIC {} max_relative={} l2_relative={} worst_index={} actual={} expected={}", stage, error,
               std::sqrt(delta / scale), worst, actual.data()[worst], expected[worst]);
  std::ofstream out(prefix + "." + stage + ".f32", std::ios::binary);
  out.write(reinterpret_cast<const char*>(actual.data()), static_cast<std::streamsize>(actual.numel() * sizeof(float)));
  require(out.good(), "diagnostic write");
}
}  // namespace
int main(int argc, char** argv) {
  const auto* models = env_or_null("UOCR_MODELS_DIR");
  if (!models) return skip("UOCR_MODELS_DIR absent");
  return run_tests([&] {
    require(argc == 2, "output prefix required");
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const CheckpointWeights weights(shards);
    const intel_native::Context ctx(0, 12ULL << 30);
    const intel_native::IntelVisionEncoder model(ctx, weights);
    const std::filesystem::path fixture = UOCR_GOLDENS "/quickstart-p2";
    std::ifstream input(fixture / "page.png", std::ios::binary);
    const std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto page = preprocess_page(decode_image(png));
    const auto sam = intel_native::IntelVisionTestAccess::sam(model, page.global);
    const auto chw = floats(fixture / "sam.f32");
    require(chw.size() == static_cast<std::size_t>(sam.numel()), "SAM reference shape");
    std::vector<float> hwc(chw.size());
    for (std::int64_t row = 0; row < sam.dim(0); ++row)
      for (std::int64_t col = 0; col < sam.dim(1); ++col) hwc[row * sam.dim(1) + col] = chw[col * sam.dim(0) + row];
    std::println("device={} native_kernels={} gpu_gemms={}", ctx.device_name(), ctx.kernel_submissions(),
                 ctx.gemm_submissions());
    report("sam", sam, hwc, argv[1]);
    report("projector", model.encode_view(page.global), floats(fixture / "projector.f32"), argv[1]);
    std::println("DIAGNOSTIC ONLY retained={} peak={}", ctx.allocated_bytes(), ctx.peak_bytes());
  });
}
