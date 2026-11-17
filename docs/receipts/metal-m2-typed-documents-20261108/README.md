# M2 typed-document RPC receipt

Qualified source: 2db7d72a473b7f064ea3ec3cc97ccf6a497b420a.
Runtime/model source identical to ab0e8f593a4dd863f3daefeae5361b649312a187;
this change imports shared document assertions and mutation tests only.
Imports: Tesla 2e73d3d/d75be71, parent bdc9ff7, NaN mutation from 245183c.
Intel-specific tests were left with their owner.

Actual Mac source directory (archive checkout, no .git):
/Users/krickert/work/astra-metal-20261108/unlimited-native-metal.
All tracked files matched source-files.sha256 before compilation. Native
WERROR targeted build passed. stb 003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5;
JSON v3.12.0 / 55f93686c01528224f448c19128836e7df245f72; no source overrides.
Homebrew gRPC 1.85.1, protobuf 36.2. pkg-config grpc reports core ABI 58.0.0.
Apple M2 arm64, macOS 27.2, Apple Clang 21; no Rosetta. All 8 original model
files freshly SHA256-verified. No converted weights.

Native and Linux host mutation tests: 11 invalid documents rejected, valid
2-page title/section-heading snapshots accepted. Native binary hashes and
complete build/configure output accompany this receipt.

Full loopback TCP model matrix PASS, exit 0: 6 alternating 1/2-page requests,
then 3 valid cancellation/admission-release/wire-recovery cycles. Every completed
request validates content, boxes, tokens, typed page/combined Document payloads,
body/caption references and page identity. Serialized document snapshots repeat
exactly across requests and after cancellation. Recovery uses actual TCP gRPC.
All 9 completed inference RPCs passed; final log ends with ok.

Execution from the directory above:
```
/usr/bin/time -l timeout -k 10 900 nice -n 10 env \
 MTL_DEBUG_LAYER=1 \
 UOCR_MODELS_DIR=/Users/krickert/work/astra-metal-20261108/models/unlimited-ocr-2 \
 GRPC_UOCR_BACKEND=metal GRPC_UOCR_WEIGHT_FORMAT=bf16 \
 GRPC_UOCR_DEVICE_MEMORY_CAP=12884901888 build/server_model_stream_test
```

Elapsed 423.53s, user 19.53s, system 27.03s. time reports maximum RSS
4775444480 bytes. This is test-process accounting, not a new device allocation
measurement or performance comparison. The timeout was unchanged at 900s.
One model instance, nice10, no concurrent model/performance job. Final process
inspection found no surviving server_model_stream_test. Mac slot explicitly
released to parent/Kant. System memory free percentage after completion: 66%.

Raw RPC output, exit status, hashes, mutation results, model/source verification,
and before/after memory reports are retained here. Performance work remains
frozen. ASR device qualification is separate and awaits reviewed fixes.
