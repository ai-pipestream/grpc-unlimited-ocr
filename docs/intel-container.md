# Intel container qualification plan and audit

Status: prepared, not image-qualified. No heavy image build or GPU image run
was performed for this change. The native receipts remain native receipts.
Pauli's LightOn clean build owns the heavy-build slot; GPU use needs a
separate explicit grant and the shared model-test lock.

## Audit against the native run

The qualified branch at c56dbed and parent integration context 9c60e6e had no
`Dockerfile.intel`. The CUDA/Jetson images cannot package this backend.
The new Intel recipe includes oneDNN as well as SYCL, OpenCL, TBB and the
Level Zero adapter/runtime libraries. Omitting oneDNN or its loader path
would leave `libdnnl.so.3` unresolved. No model files enter the image.

Native build inputs verified on krick-1:

| Input | Qualified setting / new image requirement |
|---|---|
| Compiler | IntelLLVM 2025.3.3; icx/icpx, Release `-O3 -DNDEBUG`, empty extra CXX flags, no sanitizers, WERROR off |
| LayerNorm | Source-specific `-fp-model=precise`; no blanket fast/precise FP override |
| oneDNN | 3.9.1, CPU and GPU runtimes SYCL; used only for matmul |
| gRPC | c876f4da50f7da2f331888b88b2a7243514139fe, bundled `module` SSL/BoringSSL provider |
| JSON | 55f93686c01528224f448c19128836e7df245f72 (v3.12.0) |
| stb | 003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5 |
| Level Zero / NEO | libze1 1.32.0-1; GPU/OpenCL packages 26.27.39122.14-1 |

The gRPC and JSON declarations now use their qualified full commit IDs instead
of movable tags; gRPC submodules follow that commit's gitlinks. The oneAPI base
digest is a candidate copied from the sibling recipe, not a claim that its
toolchain matches. Before configuring, `intel-qualified-toolchain.sha256`
checks the compiler, primary runtime libraries, both Level Zero adapters,
the Unified Runtime loader and Intel math libraries against the native host.
A mismatch fails closed; do not replace the hashes merely to get a build.
A different compiler/runtime requires explicit new native and image gates.

The system-OpenSSL4 workaround in LightOn is not silently imported. Native
Unlimited qualification used bundled BoringSSL. That compilation exceeded a
24 GiB builder cap in the sibling project, so this proposed build uses a
48 GiB cap with no swap and checks 52 GiB host availability before starting.
This is conservative preparation, not proof that the clean build fits.
Resource exhaustion must retain the failed log and cgroup memory events.

Ubuntu and oneAPI base digests, model hashes, and source commits are pinned.
APT's remaining OS/tool packages are inventoried in the receipts but are not
snapshot-locked. The process is repeatable; byte-identical rebuilds are not
claimed. Image IDs and binary hashes identify the artifacts actually tested.

## Clean build, only after the heavy-build slot is granted

Run from a clean committed checkout on the chosen amd64 build host, with
receipts outside the checkout. Supply a trusted digest-pinned BuildKit image.
The script creates its own builder, limits it to two CPUs and at most two
compile jobs, uses a fresh git archive and `--no-cache`, and retains logs,
source/archive hashes, CMake cache/Ninja rules, package inventories, dependency
gitlinks, binary hashes and image metadata. It stops only its own builder.
The six-hour build timeout is a failure, never a successful receipt.

```sh
export UOCR_IMAGE_BUILD_SLOT_GRANTED=1  # only after scheduler release
export BUILDKIT_IMAGE='moby/buildkit@sha256:REPLACE_WITH_VERIFIED_DIGEST'
scripts/qualify-intel-image.sh build /tmp/uocr-intel-image-build-UNIQUE
```

The `runtime` target contains the server and developer runner. The
`qualification` target extends those same layers with test clients and golden
data. Compiler caches or host-built service binaries are not copied in.
Build-time CPU tests do not initialize a GPU and cannot establish image OCR.

## Actual image entrypoint and RPC, only in a granted GPU slot

Move/load the built images on krick-1 if the build host differs. Keep the
matching clean source revision available; the launcher rejects a source/image
revision mismatch. Use the qualification tag recorded by the build script.

```sh
export UOCR_GPU_SLOT_GRANTED=1  # only after explicit B70 release
export UOCR_INTEL_TEST_LOCK=/work/main/grpc-services/worktrees/intel-sycl-repos/model-test.lock
scripts/qualify-intel-image.sh rpc /tmp/uocr-intel-image-rpc-UNIQUE \
  uocr-intel:SOURCE_SHA_PREFIX-qualification /work/models/unlimited-intel-qualification
```

This runs the image's real server entrypoint as PID 1, nonroot, read-only,
with read-only original models, a temporary cache filesystem, a 12 GiB native
allocation cap and a 32 GiB container memory cap. It verifies all eight model
hashes and free GPU headroom before startup. A separate client inside that
container performs typed GetServiceInfo and requires the actual B70 identity,
then runs the full one/two-page matrix through the entrypoint's TCP server.
It does not construct another model instance. The matrix retains the oracle,
payload/reference, exact-repeat, cancellation and +128 MiB warm RSS gates.
RSS is read from the server's `/proc/1/status`, not the client process.

The external mode cannot read the server's internal lease counters. It gives
cancellation a fixed one-second recovery grace, then requires successful new
admission; this is distinct from the native in-process lease-zero assertion.
The native gate is unchanged in scope and remains separately recorded. The
image run has a 30-minute deadline and must also stop the server cleanly with
exit 0. On failure/timeout the launcher retains server logs and container
state, stops its server, releases the lock and returns nonzero.

Broad corpus, long-document soak, and controlled speed comparisons remain
outside this matrix. No image result exists until this procedure is executed.
