#include "backend/intel/ops.h"
#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace uocr;
namespace {
void compare(const std::vector<float>& actual, const Tensor& expected, const char* name) {
  if (actual.size() != static_cast<std::size_t>(expected.numel())) throw std::runtime_error("size mismatch");
  float worst = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float error = std::fabs(actual[i] - expected.data()[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected.data()[i]) || error > 3e-5f * std::max(1.0f, std::fabs(expected.data()[i])))
      throw std::runtime_error(std::string(name) + " parity failed at " + std::to_string(i));
    worst = std::max(worst, error);
  }
  std::cout << name << " count=" << actual.size() << " max_abs=" << worst << '\n';
}
void run(intel::Context& ctx) {
  reference::set_thread_count(2);
  Tensor input({7, 129});
  for (std::int64_t i = 0; i < input.numel(); ++i) input.data()[i] = std::sin(static_cast<float>(i) * 0.173f) * 3;
  intel::Array device(ctx, static_cast<std::size_t>(input.numel()));
  for (auto op : {intel::Activation::silu, intel::Activation::gelu_erf, intel::Activation::quick_gelu}) {
    auto expected = input; device.upload(input.span());
    intel::activate(device, op);
    if (op == intel::Activation::silu) reference::silu(expected);
    else if (op == intel::Activation::gelu_erf) reference::gelu_erf(expected);
    else reference::quick_gelu(expected);
    compare(device.download(), expected, "activation");
  }
  Tensor weight({129}, 0.75f); intel::Array dw(ctx, 129); dw.upload(weight.span());
  auto expected = input; device.upload(input.span());
  intel::rms_norm(device, dw, 129, 1e-5f); reference::rms_norm(expected, weight, 1e-5f);
  compare(device.download(), expected, "rms_norm");
  expected = input;
  for (int j = 0; j < 129; ++j) expected.data()[j] = -std::numeric_limits<float>::infinity();
  device.upload(expected.span()); intel::softmax_rows(device, 129); reference::softmax_rows(expected);
  compare(device.download(), expected, "softmax_masked");
  Tensor matrix({13, 129}); Tensor bias({13}, 0.125f);
  for (std::int64_t i = 0; i < matrix.numel(); ++i) matrix.data()[i] = std::cos(static_cast<float>(i) * 0.131f) * 0.125f;
  intel::Array dm(ctx, static_cast<std::size_t>(matrix.numel())), db(ctx, 13), output(ctx, 7 * 13);
  dm.upload(matrix.span()); db.upload(bias.span()); device.upload(input.span());
  auto start = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < 100; ++repeat) intel::linear(device, dm, &db, output, 7, 13, 129);
  ctx.synchronize();
  std::cout << "linear_100_device_us=" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count() << '\n';
  compare(output.download(), reference::linear(input, matrix, &bias), "linear_bias");
  intel::linear(device, dm, nullptr, output, 7, 13, 129);
  compare(output.download(), reference::linear(input, matrix), "linear_no_bias");
  bool rejected = false;
  try { intel::softmax_rows(device, 0); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("zero width accepted");
  rejected = false;
  try { intel::Context invalid(9999); } catch (const std::runtime_error&) { rejected = true; }
  if (!rejected) throw std::runtime_error("invalid GPU accepted");
}
}
int main() {
  std::unique_ptr<intel::Context> context;
  try { context = std::make_unique<intel::Context>(); }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return std::getenv("UOCR_REQUIRE_INTEL") ? 1 : 77;
  }
  try {
    std::cout << "device=" << context->device_name() << " driver=" << context->driver_version() << '\n';
    run(*context);
    std::cout << "PASS initial Intel primitive parity; full model has separate qualification gates\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
