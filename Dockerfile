# CUDA image (tag latest). Builds the engine with the CUDA backend and runs
# the test suite in the build stage: a failing test fails the image. Tests
# that need the model weights or a GPU skip (77) here and run where those
# are (ctest with UOCR_MODELS_DIR on a GPU host).
#
#   docker build -t grpc-unlimited-ocr:local .
#
# The runtime stage carries the developer runner until the gRPC server
# lands; weights are mounted at /models, never baked in.
ARG UOCR_CUDA_ARCHITECTURES="87;89"

FROM nvidia/cuda:13.4.3-devel-ubuntu26.04 AS build
ARG UOCR_CUDA_ARCHITECTURES
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ git ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUOCR_WITH_CUDA=ON -DUOCR_WERROR=ON \
      "-DCMAKE_CUDA_ARCHITECTURES=${UOCR_CUDA_ARCHITECTURES}" \
 && cmake --build /build \
 && ctest --test-dir /build --output-on-failure \
 && mkdir -p /out && cp /build/uocr-run /out/

FROM nvidia/cuda:13.4.3-runtime-ubuntu26.04
ARG UOCR_REVISION=
LABEL org.opencontainers.image.revision=${UOCR_REVISION}
COPY --from=build /out/uocr-run /usr/local/bin/uocr-run
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/uocr-run", "--models", "/models", "--backend", "cuda"]
