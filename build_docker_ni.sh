#!/bin/bash

IMAGE=opera/dswx-ni
tag=gamma_0.3.0
echo "IMAGE is $IMAGE:$tag"

# fail on any non-zero exit codes
set -ex

python3 setup_ni.py sdist

# build image
docker build --rm --force-rm --network=host -t ${IMAGE}:$tag -f docker/Dockerfile_ni .

# create image tar
docker save opera/dswx-ni > docker/dockerimg_dswx_ni_$tag.tar

# remove image
docker image rm opera/dswx-ni:$tag    
