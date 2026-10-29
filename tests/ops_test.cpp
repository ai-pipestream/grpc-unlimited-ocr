// Reference ops against values worked by hand, and the properties the
// model relies on (identity resizes, constant preservation, masking).

#include <cmath>
#include <limits>

#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"
#include "check.h"

using namespace uocr;
namespace ref = uocr::reference;

namespace {

bool near(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b)); }

void linear_and_norms() {
  const Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
  const Tensor w({2, 3}, {1, 0, -1, 0.5f, 0.5f, 0.5f});
  const Tensor b({2}, {10, 20});
  const Tensor y = ref::linear(x, w, &b);
  require(near(y.row(0)[0], 8) && near(y.row(0)[1], 23) && near(y.row(1)[0], 8) && near(y.row(1)[1], 27.5f),
          "linear values");

  Tensor ln({1, 4}, {1, 2, 3, 4});
  ref::layer_norm(ln, Tensor({4}, 1.0f), Tensor({4}, 0.0f), 0.0f);
  const float s = std::sqrt(1.25f);
  require(near(ln.data()[0], -1.5f / s) && near(ln.data()[3], 1.5f / s), "layer norm values");

  Tensor rms({1, 2}, {3, 4});
  ref::rms_norm(rms, Tensor({2}, {2, 1}), 0.0f);
  const float r = std::sqrt(12.5f);
  require(near(rms.data()[0], 6 / r) && near(rms.data()[1], 4 / r), "rms norm values");

  Tensor chw({2, 1, 2}, {1, 3, 3, 5});  // channel values at the two positions: (1,3) and (3,5)
  ref::layer_norm_channels(chw, Tensor({2}, 1.0f), Tensor({2}, 0.0f), 0.0f);
  require(near(chw.data()[0], -1) && near(chw.data()[2], 1) && near(chw.data()[1], -1), "channel norm values");
}

void activations_and_softmax() {
  Tensor g({3}, {-1, 0, 1});
  ref::gelu_erf(g);
  require(near(g.data()[2], 0.8413447f) && near(g.data()[0], -0.1586553f) && g.data()[1] == 0, "exact gelu");
  Tensor q({1}, {1});
  ref::quick_gelu(q);
  require(near(q.data()[0], 1 / (1 + std::exp(-1.702f))), "quick gelu");
  Tensor si({1}, {2});
  ref::silu(si);
  require(near(si.data()[0], 2 / (1 + std::exp(-2.0f))), "silu");

  Tensor sm({2, 3}, {0, 0, std::log(2.0f), -std::numeric_limits<float>::infinity(),
                     -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
  ref::softmax_rows(sm);
  require(near(sm.data()[0], 0.25f) && near(sm.data()[2], 0.5f), "softmax values");
  require(sm.data()[3] == 0 && sm.data()[5] == 0, "fully masked row is zeros");
}

void conv_and_attention() {
  // 1 channel 3x3 image, 2x2 kernel of ones, stride 1, no padding: 2x2 sums.
  const Tensor img({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  const Tensor k({1, 1, 2, 2}, 1.0f);
  const Tensor y = ref::conv2d(img, k, nullptr, 1, 0);
  require(y.shape() == std::vector<std::int64_t>{1, 2, 2} && near(y.data()[0], 12) && near(y.data()[3], 28),
          "conv2d sums");
  const Tensor yp = ref::conv2d(img, Tensor({1, 1, 3, 3}, 1.0f), nullptr, 2, 1);  // stride 2, pad 1
  require(yp.shape() == std::vector<std::int64_t>{1, 2, 2} && near(yp.data()[0], 12) && near(yp.data()[3], 28),
          "conv2d stride and padding");

  // Two keys; the bias masks the second, so the output is the first value.
  const Tensor qv({1, 2}, {1, 0});
  const Tensor kv({2, 2}, {1, 0, 0, 1});
  const Tensor vv({2, 2}, {5, 6, 7, 8});
  const Tensor mask({1, 2}, {0, -std::numeric_limits<float>::infinity()});
  const Tensor a = ref::attention_head(qv, kv, vv, &mask, 1.0f);
  require(near(a.data()[0], 5) && near(a.data()[1], 6), "masked attention");
  const Tensor even = ref::attention_head(Tensor({1, 2}, 0.0f), kv, vv, nullptr, 1.0f);
  require(near(even.data()[0], 6) && near(even.data()[1], 7), "uniform attention averages");
}

void resizes() {
  Tensor ramp({1, 6, 6});
  for (std::int64_t i = 0; i < 36; ++i) ramp.data()[i] = static_cast<float>(i);
  const Tensor same = ref::resize_bicubic_aa(ramp, 6, 6);
  for (std::int64_t i = 0; i < 36; ++i) require(near(same.data()[i], ramp.data()[i]), "same-size bicubic is identity");

  const Tensor flat = ref::resize_bicubic_aa(Tensor({2, 16, 16}, 3.5f), 10, 10);
  for (const float v : flat.span()) require(near(v, 3.5f), "antialiased downscale keeps a constant");

  // Away from the edges a linear ramp lands within a few thousandths of its value at
  // each output centre: the scaled Keys kernel is not exactly linear-preserving at
  // non-integer tap spacing, which is a property of the algorithm, not of this port.
  Tensor xr({1, 1, 64});
  for (std::int64_t i = 0; i < 64; ++i) xr.data()[i] = static_cast<float>(i);
  const Tensor down = ref::resize_bicubic_aa(xr, 1, 40);
  const float scale = 64.0f / 40.0f;
  for (std::int64_t i = 4; i < 36; ++i) {
    require(std::fabs(down.data()[i] - (scale * (static_cast<float>(i) + 0.5f) - 0.5f)) < 5e-3f,
            "antialiased downscale tracks a ramp");
  }

  const Tensor two({2, 1}, {0, 4});
  const Tensor four = ref::resize_linear_rows(two, 4);
  require(near(four.data()[0], 0) && near(four.data()[1], 1) && near(four.data()[2], 3) && near(four.data()[3], 4),
          "linear upsample matches align_corners=False");
}

void thread_independence() {
  Tensor x({33, 70}), w({17, 70});
  for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] = std::sin(static_cast<float>(i));
  for (std::int64_t i = 0; i < w.numel(); ++i) w.data()[i] = std::cos(static_cast<float>(i) * 0.7f);
  ref::set_thread_count(1);
  const Tensor serial = ref::linear(x, w);
  ref::set_thread_count(7);
  const Tensor threaded = ref::linear(x, w);
  ref::set_thread_count(std::thread::hardware_concurrency());
  for (std::int64_t i = 0; i < serial.numel(); ++i) {
    require(serial.data()[i] == threaded.data()[i], "linear is bitwise identical across thread counts");
  }
}

}  // namespace

int main() {
  return run_tests([] {
    linear_and_norms();
    activations_and_softmax();
    conv_and_attention();
    resizes();
    thread_independence();
  });
}
