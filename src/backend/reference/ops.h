#pragma once

// The reference backend's op set: plain fp32 C++, written for clarity and
// determinism, used as the oracle every fast backend is tested against.
// Shapes follow PyTorch's conventions for the same op so ports stay
// line-for-line comparable with the upstream model code.

#include <cstdint>
#include <optional>

#include "tensor/tensor.h"

namespace uocr::reference {

// x [M, K] times weight [N, K] transposed, plus bias [N]: [M, N].
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr);

// Normalises each row of x (last axis) in place.
void layer_norm(Tensor& x, const Tensor& weight, const Tensor& bias, float eps);
void rms_norm(Tensor& x, const Tensor& weight, float eps);

// Normalises over channels at each spatial position of x [C, H, W].
void layer_norm_channels(Tensor& x, const Tensor& weight, const Tensor& bias, float eps);

// Elementwise activations, in place.
void gelu_erf(Tensor& x);    // x * Phi(x), the exact GELU
void quick_gelu(Tensor& x);  // x * sigmoid(1.702 x)
void silu(Tensor& x);        // x * sigmoid(x)

// Softmax over the last axis, in place.
void softmax_rows(Tensor& x);

// x [C, H, W], weight [O, C, KH, KW], optional bias [O]: [O, HO, WO].
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor* bias, int stride, int padding);

// One attention head: q [Lq, D], k and v [Lk, D], optional additive bias
// [Lq, Lk]; softmax(q k^T * scale + bias) v: [Lq, D].
Tensor attention_head(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor* bias, float scale);

// Resizes x [C, H, W] to [C, OH, OW] like
// F.interpolate(mode="bicubic", antialias=True, align_corners=False).
Tensor resize_bicubic_aa(const Tensor& x, std::int64_t out_h, std::int64_t out_w);

// Resizes each column of x [L, C] to [OL, C] like F.interpolate over L with
// mode="linear", align_corners=False.
Tensor resize_linear_rows(const Tensor& x, std::int64_t out_len);

// Permutes a [A, B, C] tensor to [B, C, A] or back: the HWC/CHW moves the
// vision encoder makes.
Tensor chw_to_hwc(const Tensor& x);
Tensor hwc_to_chw(const Tensor& x);

}  // namespace uocr::reference
