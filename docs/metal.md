# Native Metal backend

Build on native Apple Silicon with C++23, the macOS Metal/MPS frameworks,
CMake and an installed native gRPC/protobuf toolchain:

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DUOCR_WITH_METAL=ON -DUOCR_SYSTEM_GRPC=ON \
  -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build-metal -j2
ctest --test-dir build-metal -R '^metal_(ops|vision|decoder)_test$' --output-on-failure
```

`UOCR_SYSTEM_GRPC` is opt-in. The default server build still fetches the
repository's pinned gRPC. Native qualification records the installed
version separately; it does not imply CI used that version.

The backend runs SAM, CLIP, the projector, decoder attention, page2d RoPE,
block summaries/entmax selection and MoE arithmetic through native Metal
compute. MPS is used only for dense matrix multiplication. Decode GEMV
and routed experts read bf16 weight bits directly in our SIMD kernels.
MPS prefill GEMM widens weights into temporary fp32 buffers on the GPU.
Activations and accumulation remain fp32. No inference framework, graph
runtime, Python or converted weight files are used.

Model weights come from the manifest-pinned upstream safetensors files.
The existing loader maps the shards; this implementation copies bf16
weights into shared Metal buffers. It does not yet import the mappings
without copies. The model's stored bf16 values are preserved exactly.
Image decoding/letterboxing, prompt assembly, page bookkeeping, prefill
expert grouping metadata and generation policy remain on the host. Learned
position interpolation and all neural-network arithmetic run on Metal.
Commands are encoded per step; there is no indirect-command-buffer replay.

Select the backend explicitly:

```sh
GRPC_UOCR_BACKEND=metal GRPC_UOCR_WEIGHT_FORMAT=bf16 \
GRPC_UOCR_MODELS_DIR=/path/to/unlimited-ocr-2 \
  ./build-metal/grpc-unlimited-ocr-server
```

An unavailable device or unsupported weight format fails startup. There is
no automatic reference/CPU fallback. `GetServiceInfo` names the selected
Metal device. The context caps its owned live buffers at the smaller of
`GRPC_UOCR_DEVICE_MEMORY_CAP` (12 GiB when unset) and the device's recommended working set; MPS internal resources,
the checkpoint mappings and host tensors are outside that owned-buffer
accounting. Process RSS must therefore be measured separately.

Qualification gates:

- `metal_ops_test`: primitive comparisons against the reference, including
  routing masks, page2d RoPE, convolution, attention products and norms.
- `metal_vision_test`: a small tiled SAM/CLIP model; with `UOCR_MODELS_DIR`,
  also the real page's projector features against the recorded upstream output.
- `metal_decoder_test`: reference logits after prefill and 44 steps through
  dense/MoE layers, including page transitions and remote blocks.
- `metal_golden_test`: the known real page twice, exact text/labels,
  coordinates within 2/1000 and token-identical repeated output. It records
  timings, process RSS, owned-buffer peaks and native command counts.
- `server_model_stream_test`: real-model RPC tests; set
  `GRPC_UOCR_BACKEND=metal` and `GRPC_UOCR_WEIGHT_FORMAT=bf16` explicitly.

Tests requiring weights exit 77 when the model directory is missing.
A native build and the tiny parity tests do not qualify real OCR, RPC
recovery, memory fit or speed. See the qualification receipt for measured
results and remaining gaps.

The developer runner bounds its reference CPU backend to two threads by
default, including when used for the same-Mac baseline.
