#!/usr/bin/env bash

set -euo pipefail

IMAGE="${IMAGE:-opera/dswx-ni}"
TAG="${TAG:-calval_0.4.2}"
PLATFORM="linux/amd64"
OUTPUT_TAR="docker/dockerimg_dswx_ni_${TAG}.tar"

echo "Building ${IMAGE}:${TAG} for ${PLATFORM}"

docker buildx build \
    --platform "${PLATFORM}" \
    --load \
    --progress=plain \
    --tag "${IMAGE}:${TAG}" \
    --file docker/Dockerfile_ni \
    .

IMAGE_OS="$(
    docker image inspect \
        --format '{{.Os}}' \
        "${IMAGE}:${TAG}"
)"

IMAGE_ARCH="$(
    docker image inspect \
        --format '{{.Architecture}}' \
        "${IMAGE}:${TAG}"
)"

if [[ "${IMAGE_OS}/${IMAGE_ARCH}" != "${PLATFORM}" ]]; then
    echo "ERROR: Expected ${PLATFORM}, got ${IMAGE_OS}/${IMAGE_ARCH}"
    exit 1
fi

echo "Saving ${IMAGE}:${TAG} to ${OUTPUT_TAR}"

docker save \
    --output "${OUTPUT_TAR}" \
    "${IMAGE}:${TAG}"

echo "Successfully created ${OUTPUT_TAR}"