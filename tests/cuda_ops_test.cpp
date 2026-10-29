// Every CUDA op against its reference-backend counterpart on random data.
// The bf16 GEMM is judged at bf16 tolerance, the rest at fp32 tolerance.
// Skips (77) without a CUDA device.

#include <cmath>
#include <random>

#include "backend/cuda/context.h"
#include "backend/cuda/ops.h"
#include "backend/reference/ops.h"
#include "check.h"
#include "model/attention_plan.h"
#include "model/block_router.h"
#include "model/decoder.h"
#include "model/page_rope.h"
#include "tensor/dtype.h"

using namespace uocr;
namespace cu = uocr::cuda;
namespace ref = uocr::reference;

namespace {

std::mt19937 rng(5);

Tensor random(std::vector<std::int64_t> shape, float scale = 1.0f) {
  std::normal_distribution<float> n(0.0f, scale);
  Tensor t(std::move(shape));
  for (float& v : t.span()) v = n(rng);
  return t;
}

cu::Array up(const cu::Context& ctx, const Tensor& t, cu::DType type = cu::DType::kF32) {
  cu::Array a(ctx, type, t.numel());
  a.upload(t.data(), t.numel());
  return a;
}

cu::Array up_ids(const cu::Context& ctx, const std::vector<std::int64_t>& ids) {
  cu::Array a(ctx, cu::DType::kI64, static_cast<std::int64_t>(ids.size()));
  a.upload(ids.data(), static_cast<std::int64_t>(ids.size()));
  return a;
}

double max_rel(const std::vector<float>& got, const Tensor& want) {
  double worst = 0.0, scale = 1e-9;
  for (std::int64_t i = 0; i < want.numel(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(got[static_cast<std::size_t>(i)]) - want.data()[i]));
    scale = std::max(scale, std::fabs(static_cast<double>(want.data()[i])));
  }
  return worst / scale;
}

void expect(const char* what, double err, double tol) {
  std::println("{:<22} rel err {:.2e} (tol {:.0e})", what, err, tol);
  require(err <= tol, what);
}

Tensor bf16_rounded(Tensor t) {
  for (float& v : t.span()) v = bf16_to_f32(f32_to_bf16(v));
  return t;
}

}  // namespace

int main() {
  if (cu::device_count() == 0) return skip("no CUDA device");
  return run_tests([] {
    const cu::Context ctx(0);

    {  // bf16 linear with bias, against the reference on bf16-rounded inputs
      const Tensor x = random({37, 96}), w = random({50, 96}, 0.1f), b = random({50});
      const Tensor want = ref::linear(bf16_rounded(x), bf16_rounded(w), &b);
      auto dx = up(ctx, x), dw = up(ctx, w, cu::DType::kBF16), db = up(ctx, b);
      cu::Array dy(ctx, cu::DType::kF32, 37 * 50);
      cu::linear(ctx, dx.f32(), 37, 96, dw.bf16(), 50, db.f32(), dy.f32());
      expect("linear bf16", max_rel(dy.download(), want), 1e-5);
    }
    {  // fp32 batched products
      const Tensor a = random({2 * 5 * 8}), b = random({2 * 7 * 8});
      cu::Array da = up(ctx, a), db = up(ctx, b), dc(ctx, cu::DType::kF32, 2 * 5 * 7);
      cu::gemm_nt(ctx, 5, 7, 8, 0.5f, da.f32(), 8, 40, db.f32(), 8, 56, dc.f32(), 7, 35, 2);
      const auto got = dc.download();
      double worst = 0.0;
      for (int bt = 0; bt < 2; ++bt)
        for (int i = 0; i < 5; ++i)
          for (int j = 0; j < 7; ++j) {
            float s = 0.0f;
            for (int e = 0; e < 8; ++e) s += a.data()[bt * 40 + i * 8 + e] * b.data()[bt * 56 + j * 8 + e];
            worst = std::max(worst, std::fabs(static_cast<double>(got[bt * 35 + i * 7 + j]) - 0.5 * s));
          }
      expect("gemm_nt batched", worst, 1e-5);
      const Tensor p = random({5, 7}), v = random({7, 3});
      cu::Array dp = up(ctx, p), dv = up(ctx, v), dout(ctx, cu::DType::kF32, 15);
      cu::gemm_nn(ctx, 5, 3, 7, dp.f32(), 7, 0, dv.f32(), 3, 0, dout.f32(), 3, 0, 1);
      Tensor vt({3, 7});
      for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 3; ++j) vt.row(j)[i] = v.row(i)[j];
      expect("gemm_nn", max_rel(dout.download(), ref::linear(p, vt)), 1e-5);
    }
    {  // norms and activations
      const Tensor x = random({9, 300}), w = random({300}), b = random({300});
      Tensor want = x;
      ref::layer_norm(want, w, b, 1e-6f);
      auto dx = up(ctx, x), dw = up(ctx, w), db = up(ctx, b);
      cu::layer_norm(ctx, dx.f32(), dw.f32(), db.f32(), 9, 300, 1e-6f);
      expect("layer_norm", max_rel(dx.download(), want), 1e-5);
      Tensor rwant = x;
      ref::rms_norm(rwant, w, 1e-6f);
      auto dr = up(ctx, x);
      cu::rms_norm(ctx, dr.f32(), dw.f32(), 9, 300, 1e-6f);
      expect("rms_norm", max_rel(dr.download(), rwant), 1e-5);
      Tensor g = x;
      ref::gelu_erf(g);
      auto dg = up(ctx, x);
      cu::gelu_erf(ctx, dg.f32(), x.numel());
      expect("gelu_erf", max_rel(dg.download(), g), 1e-6);
      Tensor qg = x;
      ref::quick_gelu(qg);
      auto dq = up(ctx, x);
      cu::quick_gelu(ctx, dq.f32(), x.numel());
      expect("quick_gelu", max_rel(dq.download(), qg), 1e-6);
      Tensor sm = x;
      ref::softmax_rows(sm);
      auto ds = up(ctx, x);
      cu::softmax_rows(ctx, ds.f32(), 9, 300);
      expect("softmax_rows", max_rel(ds.download(), sm), 1e-5);
    }
    {  // convolution through im2col + linear, and the layout moves
      const Tensor img = random({3, 32, 32}), w = random({8, 3, 3, 3}, 0.2f);
      const Tensor want = ref::conv2d(bf16_rounded(img), bf16_rounded(w), nullptr, 2, 1);  // [8, 16, 16]
      auto dimg = up(ctx, img), dw = up(ctx, w, cu::DType::kBF16);
      cu::Array cols(ctx, cu::DType::kF32, 16 * 16 * 27), hwc(ctx, cu::DType::kF32, 16 * 16 * 8),
          chw(ctx, cu::DType::kF32, 8 * 16 * 16);
      cu::im2col(ctx, dimg.f32(), 3, 32, 32, 3, 3, 2, 1, cols.f32());
      cu::linear(ctx, cols.f32(), 256, 27, dw.bf16(), 8, nullptr, hwc.f32());
      cu::hwc_to_chw(ctx, hwc.f32(), 16, 16, 8, chw.f32());
      expect("conv2d via im2col", max_rel(chw.download(), want), 1e-5);
    }
    {  // rope at page coordinates
      PageRope rope(16);
      Tensor x = random({4, 32});
      const std::vector<RopeCoord> coords{{0, 0}, {5, 1}, {300, 2}, {7, 3}};
      Tensor want = x;
      rope.apply(want, coords);
      std::vector<std::int64_t> tok, page;
      for (const auto& c : coords) {
        tok.push_back(c.tok);
        page.push_back(c.page);
      }
      std::vector<float> inv_tok, inv_page{1.0f};
      for (int i = 0; i < 7; ++i) inv_tok.push_back(static_cast<float>(1.0 / std::pow(1e4, i / 7.0)));
      auto dx = up(ctx, x), dt = up_ids(ctx, tok), dp = up_ids(ctx, page);
      cu::Array dit(ctx, cu::DType::kF32, 7), dip(ctx, cu::DType::kF32, 1);
      dit.upload(inv_tok.data(), 7);
      dip.upload(inv_page.data(), 1);
      cu::rope(ctx, dx.f32(), 4, 2, 16, dt.i64(),
               dp.i64(), dit.f32(), 7, dip.f32(), 1, 2);
      expect("rope page2d", max_rel(dx.download(), want), 1e-5);
    }
    {  // gather, scatter-add, router summaries
      const Tensor src = random({6, 10});
      const std::vector<std::int64_t> idx{4, 0, 4};
      auto ds = up(ctx, src), di = up_ids(ctx, idx);
      cu::Array dst(ctx, cu::DType::kF32, 30);
      cu::gather_rows(ctx, ds.f32(), di.i64(), 3, 10, dst.f32());
      const auto g = dst.download();
      require(g[0] == src.row(4)[0] && g[10] == src.row(0)[0] && g[29] == src.row(4)[9], "gather_rows");
      const Tensor wts({3}, {0.5f, 2.0f, 1.0f});
      auto dwts = up(ctx, wts);
      cu::Array acc(ctx, cu::DType::kF32, 60);
      acc.zero();
      cu::scatter_add_rows(ctx, acc.f32(), di.i64(), dwts.f32(), dst.f32(), 3, 10);
      const auto s = acc.download();
      require(std::fabs(s[40] - 1.5f * src.row(4)[0]) < 1e-5f && std::fabs(s[0] - 2.0f * src.row(0)[0]) < 1e-5f,
              "scatter_add_rows");

      const Tensor keys = random({70, 16}), head = random({16});
      const Tensor want = block_summaries(keys, head, 32);
      auto dk = up(ctx, keys), dh = up(ctx, head);
      cu::Array out(ctx, cu::DType::kF32, 3 * 16);
      cu::block_summaries(ctx, dk.f32(), 70, 16, dh.f32(), 32, out.f32());
      expect("block_summaries", max_rel(out.download(), want), 1e-5);
    }
    {  // device block routing against the host selector and key mask
      RouterParams rp = Decoder::default_router();
      rp.block_size = 4;
      rp.local_blocks = 2;
      rp.top_m = 6;
      rp.top_k = 4;
      const std::int64_t rows = 5, first = 40, keys = first + rows, nblk = (keys + 3) / 4;
      Tensor z = random({rows, nblk});
      // Two tied scores, to pin the tie rule.
      z.row(0)[1] = z.row(0)[3] = 2.5f;
      BlockPages blocks{std::vector<std::int64_t>(static_cast<std::size_t>(nblk), -1),
                        std::vector<std::int64_t>(static_cast<std::size_t>(nblk), -1)};
      for (std::int64_t b = 0; b < 8; ++b) {
        blocks.min_page[static_cast<std::size_t>(b)] = b < 4 ? 0 : 1;
        blocks.max_page[static_cast<std::size_t>(b)] = b < 4 ? 0 : 1;
      }
      const std::vector<std::int64_t> pages{0, 1, -1, 5, 1};  // page 5 has no block: unrestricted
      std::vector<std::vector<std::uint8_t>> cands;
      for (const auto pg : pages) cands.push_back(page_candidates(blocks, pg, nblk));
      Tensor masked = z;
      mask_block_scores(masked, first, rp);
      const auto want = key_mask(select_blocks(masked, flatten_candidates(cands, nblk), rp), rows, first, keys, rp);

      auto dz = up(ctx, z), dmin = up_ids(ctx, blocks.min_page), dmax = up_ids(ctx, blocks.max_page),
           dpage = up_ids(ctx, pages);
      cu::Array dmask(ctx, cu::DType::kF32, rows * keys);
      cu::route_blocks(ctx, dz.f32(), rows, nblk, first, keys, dmin.i64(), dmax.i64(), nblk, dpage.i64(), -1,
                       {rp.block_size, rp.local_blocks, rp.top_m, rp.top_k, rp.idx_scale, rp.floor, rp.sigma},
                       dmask.f32());
      const auto got = dmask.download();
      double worst = 0.0;
      for (std::size_t i = 0; i < want.size(); ++i) {
        const bool a = std::isinf(want[i]), b = std::isinf(got[i]);
        require(a == b, "device routing reads the same keys as the host plan");
        if (!a) worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - want[i]));
      }
      expect("route_blocks bias", worst, 1e-5);
    }
  });
}
