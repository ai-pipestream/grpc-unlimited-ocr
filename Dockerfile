# syntax=docker/dockerfile:1.28
# CUDA image (tag latest). Builds the engine with the CUDA backend and the
# gRPC server, and runs the test suite in the build stage: a failing test
# fails the image. Tests that need the model weights or a GPU skip (77)
# here and run where those are (ctest with UOCR_MODELS_DIR on a GPU host).
#
#   docker build -t grpc-unlimited-ocr:local .
#   docker run --rm --gpus all -p 50073:50073 \
#     -v /path/to/unlimited-ocr-2:/models:ro grpc-unlimited-ocr:local
#
# The server listens on 50073 (gRPC only). Weights are mounted read-only at
# /models, never baked in; nothing is written while parsing, so the
# container runs with --read-only. The developer runner uocr-run stays in
# the image.
ARG UOCR_CUDA_ARCHITECTURES="87;89"

FROM nvidia/cuda:13.4.3-devel-ubuntu26.04 AS build
ARG UOCR_CUDA_ARCHITECTURES
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ git ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# gRPC builds from source. The cache id names every ABI-sensitive input;
# bump it when gRPC, CUDA, the base image or the toolchain moves.
ARG TARGETARCH
RUN --mount=type=cache,id=grpc-unlimited-ocr-cuda13.4-ubuntu26-grpc1.83.0-${TARGETARCH},target=/build \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUOCR_WITH_CUDA=ON -DUOCR_WERROR=ON \
      "-DCMAKE_CUDA_ARCHITECTURES=${UOCR_CUDA_ARCHITECTURES}" \
 && cmake --build /build \
 && ctest --test-dir /build --output-on-failure \
 && mkdir -p /out && cp /build/uocr-run /build/grpc-unlimited-ocr-server /out/

FROM nvidia/cuda:13.4.3-runtime-ubuntu26.04
ARG UOCR_REVISION=
LABEL org.opencontainers.image.revision=${UOCR_REVISION}
COPY --from=build /out/uocr-run /usr/local/bin/uocr-run
COPY --from=build /out/grpc-unlimited-ocr-server /usr/local/bin/grpc-unlimited-ocr-server
ENV GRPC_UOCR_LISTEN_ADDRESS=0.0.0.0:50073 \
    GRPC_UOCR_BACKEND=cuda \
    GRPC_UOCR_MODELS_DIR=/models
USER 65532:65532
EXPOSE 50073
ENTRYPOINT ["/usr/local/bin/grpc-unlimited-ocr-server"]
