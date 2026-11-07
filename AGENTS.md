# AGENTS.md: grpc-unlimited-ocr

A gRPC collector in the ai-pipestream parser fleet. It runs the
Unlimited-OCR 2 model on its own native engine and streams typed results
to gRParse, which dials it as `GRPARSE_UNLIMITED_OCR_TARGET`.

## Read this first, in order

1. This file
2. `docs/design.md`: why it is a separate service, the engine and backend
   layout, the wire contract, phases and gates
3. `../gRParse/AGENTS.md`: the fleet rules (diskless, both remotes, typed
   protos, exit codes over greps)

## Rules for this repo

- **No inference runtime and no Python.** The forward pass is ours; vendor
  libraries are allowed only for dense matrix multiply (cuBLASLt, oneDNN,
  MPS). The single Python touch is outside the repo: goldens recorded once
  from upstream's reference and committed as data.
- **Weights are the upstream files as published,** pinned by sha256 in
  `models/MANIFEST`. Never convert, derive or commit weights.
- **The reference backend is the oracle.** Every fast backend kernel is
  tested against it. A fast backend never changes output without a
  reference-backend test showing why.
- **Typed wire only.** No JSON strings, maps or blobs on the gRPC contract.
  JSON is read only from model files at startup: the safetensors header,
  the shard index and `tokenizer.json`.
- **C++23, no god classes:** files under about 800 lines, functions under
  about 60 unless they are a flat table.
- **Tests:** plain-main executables under `tests/`, registered with CTest;
  a test that needs weights or a device exits 77 when it is absent. CUDA
  tests judge at bf16 tolerance against the reference backend; the golden
  page is token-exact on the reference backend and text-exact on CUDA.
- **CUDA build with a CUDA 12 toolkit:** `cmake -B build-cuda -DUOCR_WITH_CUDA=ON
  -DCMAKE_CUDA_HOST_COMPILER=g++-13` (nvcc 12 cannot use GCC 15 as its
  host compiler; the CUDA 13 image can).
- **Git:** `origin` = Forgejo (`git.rokkon.com/ai-pipestream`), `github` =
  GitHub. Push `origin` first, then `github`. No em dashes and no tool or
  model attribution in commit messages; docs-only commits carry
  `[ci skip]`.

## Where things are

| Path | What |
|---|---|
| `src/tensor/` | element types and conversions |
| `src/weights/` | safetensors mapping, the shard index, the architecture table |
| `src/backend/reference/` | the fp32 oracle backend: ops and the thread splitter |
| `src/model/` | SAM, CLIP, projector, page layout, page rope, block router, MoE, decoder |
| `src/runtime/` | prompt, logit policy (n-gram ban, page guard), generation loop |
| `src/image/` | PNG/JPEG decode, Pillow-exact bicubic, letterbox and tiles |
| `src/tokenizer/` | byte-level BPE decoding and special ids |
| `src/backend/cuda/` | CUDA backend: context, device arrays, graphs, the op set (`ops.cu`) and the length-independent decode-step kernels (`decode_ops.cu`) |
| `src/model/cuda/` | the vision encoder and decoder on the CUDA backend; the decode step is captured as one CUDA graph |
| `proto/` | the wire contract (`unlimitedocr/v1`) and the vendored fleet `document.proto` (never edit the vendored copies) |
| `src/output/` | the block grammar ported from upstream, the incremental block parser, HTML tables, continuation links, the Document fold |
| `src/service/` | configuration, the `Engine` interface and the real engine behind it, page intake and budgets, one document's session, the gRPC handlers |
| `src/server_main.cpp` | the server (`grpc-unlimited-ocr-server`, the image entrypoint) |
| `tools/uocr_run.cpp` | developer runner: page files in, raw model output out |
| `models/MANIFEST` | the pinned upstream files |
| `docs/design.md` | the design; keep it in step with the code |
