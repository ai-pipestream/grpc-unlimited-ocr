#!/usr/bin/env bash
# Scheduling grants are separate from locks. No GPU during `build`.
set -euo pipefail
mode=${1:?usage: qualify-intel-image.sh build|rpc RECEIPTS [IMAGE MODELS]}
receipts=$(realpath -m "${2:?receipt directory required}")
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$receipts"
exec > >(tee -a "$receipts/session.log") 2>&1
trap 'result=$?; printf "IMAGE_GATE mode=%s exit=%s utc=%s\n" "$mode" "$result" "$(date -u +%FT%TZ)"' EXIT
date -u +%FT%TZ
uname -a
docker version
# Disk/memory checks and model bind mounts refer to this host. Do not silently
# send a scheduled build or GPU run to a different daemon via Docker context.
daemon_endpoint=${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}
[[ $daemon_endpoint == unix://* ]] || { echo 'Run on the scheduled Linux host with its local Docker daemon'; exit 64; }
case "$mode" in
  build)
    : "${UOCR_IMAGE_BUILD_SLOT_GRANTED:?set only after the shared heavy-build slot is granted}"
    [[ $UOCR_IMAGE_BUILD_SLOT_GRANTED == 1 ]]
    : "${BUILDKIT_IMAGE:?supply a digest-pinned BuildKit image}"
    [[ $BUILDKIT_IMAGE == *@sha256:* ]]
    [[ $(uname -m) == x86_64 ]]
    [[ -z $(git -C "$repo" status --porcelain) ]]
    exec 8>"${UOCR_IMAGE_BUILD_LOCK:-/tmp/grpc-intel-image-build.lock}"
    flock -n 8
    # Bundled BoringSSL previously exceeded 24 GiB in another Intel build.
    # Keep its qualified provider, reserve headroom, and fail rather than swap.
    available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    (( available >= 52 * 1024 * 1024 )) || { echo 'Need 52 GiB available for the 48 GiB builder cap'; exit 75; }
    revision=$(git -C "$repo" rev-parse HEAD)
    printf '%s\n' "$revision" > "$receipts/source.sha"
    git -C "$repo" archive --format=tar HEAD > "$receipts/source.tar"
    sha256sum "$receipts/source.tar"
    context=$(mktemp -d "$receipts/context.XXXXXX")
    tar -xf "$receipts/source.tar" -C "$context"
    builder="uocr-intel-${revision:0:12}-$$"
    build_cleanup() {
      result=$?
      docker inspect "buildx_buildkit_${builder}0" > "$receipts/builder-container.json" 2>/dev/null || true
      docker exec "buildx_buildkit_${builder}0" cat /sys/fs/cgroup/memory.events > "$receipts/builder-memory-events.txt" 2>/dev/null || true
      docker buildx stop "$builder" >/dev/null 2>&1 || true
      printf 'IMAGE_BUILD_EXIT=%s UTC=%s\n' "$result" "$(date -u +%FT%TZ)"
      exit "$result"
    }
    trap build_cleanup EXIT
    docker buildx create --name "$builder" --driver docker-container \
      --driver-opt "image=$BUILDKIT_IMAGE,memory=48g,memory-swap=48g,cpu-period=100000,cpu-quota=200000" \
      > "$receipts/builder-name.txt"
    # Keep builder/cache on failure for inspection; never stop someone else's builder.
    docker buildx inspect "$builder" --bootstrap > "$receipts/builder.txt"
    tag="uocr-intel:${revision:0:12}"
    timeout --kill-after=30s 21600s docker buildx build --builder "$builder" --platform linux/amd64 \
      --no-cache --progress plain --load -f "$context/Dockerfile.intel" --target qualification \
      --build-arg BUILD_JOBS=2 --build-arg "UOCR_REVISION=$revision" \
      --metadata-file "$receipts/build-metadata.json" -t "$tag-qualification" "$context"
    timeout --kill-after=30s 1800s docker buildx build --builder "$builder" --platform linux/amd64 \
      --progress plain --load -f "$context/Dockerfile.intel" --target runtime \
      --build-arg BUILD_JOBS=2 --build-arg "UOCR_REVISION=$revision" -t "$tag" "$context"
    docker image inspect "$tag" "$tag-qualification" > "$receipts/images.json"
    for image in "$tag" "$tag-qualification"; do
      docker run --rm --network none --cpus 2 --memory 1g --entrypoint /bin/sh "$image" -ec \
        'sha256sum /usr/local/bin/grpc-unlimited-ocr-server; ldd /usr/local/bin/grpc-unlimited-ocr-server > /tmp/ldd.txt; cat /tmp/ldd.txt; ! grep -q "not found" /tmp/ldd.txt; dpkg-query -W'
    done
    cid=$(docker create "$tag-qualification")
    docker cp "$cid:/usr/share/uocr/receipts" "$receipts/image-build"
    docker rm "$cid"
    printf '%s\n' "$tag-qualification" > "$receipts/qualification-image.txt"
    docker buildx stop "$builder"
    echo 'BUILD ONLY: GPU boot and RPC are NOT qualified.'
    ;;
  rpc)
    : "${UOCR_GPU_SLOT_GRANTED:?set only after explicit B70 release/grant}"
    [[ $UOCR_GPU_SLOT_GRANTED == 1 ]]
    : "${UOCR_INTEL_TEST_LOCK:?shared model-test.lock required}"
    exec 9>"$UOCR_INTEL_TEST_LOCK"
    flock -n 9
    image=${3:?qualification image required}
    models=$(realpath "${4:?original model directory required}")
    image_id=$(docker image inspect --format '{{.Id}}' "$image")
    docker image inspect "$image_id" > "$receipts/tested-image.json"
    revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image_id")
    [[ $revision == "$(git -C "$repo" rev-parse HEAD)" ]] || { echo 'Image/source revision mismatch'; exit 1; }
    [[ -z $(git -C "$repo" status --porcelain) ]]
    git -C "$repo" rev-parse HEAD
    (cd "$models"; awk '!/^#/ && NF>=3 {print $2 "  " $1}' "$repo/models/MANIFEST" | sha256sum -c -)
    awk '!/^#/ && NF>=3 {print $2 "  " $1}' "$repo/models/MANIFEST"
    groups=()
    for node in /dev/dri/renderD*; do groups+=(--group-add "$(stat -c %g "$node")"); done
    name="uocr-intel-qualification-$$"
    cleanup() {
      result=$?
      docker logs "$name" > "$receipts/server.log" 2>&1 || true
      docker inspect "$name" > "$receipts/server-container.json" 2>/dev/null || true
      docker stop --time 15 "$name" >/dev/null 2>&1 || true
      docker rm "$name" >/dev/null 2>&1 || true
      printf 'IMAGE_RPC_EXIT=%s UTC=%s\n' "$result" "$(date -u +%FT%TZ)"
      exit "$result"
    }
    trap cleanup EXIT
    # Read-only Sysman preflight, no model instance.
    docker run --rm --device /dev/dri "${groups[@]}" --entrypoint /usr/local/bin/intel_memory_check "$image_id"
    docker run -d --name "$name" --device /dev/dri "${groups[@]}" --read-only \
      --memory 32g --memory-swap 32g --tmpfs /tmp:rw,nosuid,size=1g \
      --mount "type=bind,src=$models,dst=/models,readonly" \
      -e GRPC_UOCR_PAGE_WINDOW=2 -e GRPC_UOCR_MAX_PAGES=2 -e GRPC_UOCR_MAX_TOKENS_PER_PAGE=600 "$image_id"
    docker exec "$name" sha256sum /usr/local/bin/grpc-unlimited-ocr-server /usr/local/bin/intel_native_rpc_test
    # Client runs in this server's PID namespace; /proc/1/status is SERVER RSS.
    timeout --kill-after=10s 150s docker exec -e 'UOCR_REQUIRED_DEVICE=Intel(R) Arc(TM) Pro B70 Graphics' "$name" /usr/local/bin/intel_native_rpc_test \
      --endpoint 127.0.0.1:50073 --info-only
    timeout --kill-after=10s 1800s docker exec -e 'UOCR_REQUIRED_DEVICE=Intel(R) Arc(TM) Pro B70 Graphics' "$name" /usr/local/bin/intel_native_rpc_test \
      --endpoint 127.0.0.1:50073
    docker stop --time 15 "$name"
    [[ $(docker inspect --format '{{.State.ExitCode}}' "$name") == 0 ]]
    ;;
  *) exit 64;;
esac
