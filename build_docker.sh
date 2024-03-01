#!/bin/bash

IMAGE=opera/dswx-s1
tag=calval_0.4
echo "IMAGE is $IMAGE:$tag"

# fail on any non-zero exit codes
set -ex

python3 setup.py sdist

# build image
docker build --rm --force-rm --network=host -t ${IMAGE}:$tag -f docker/Dockerfile .

# create image tar
docker save opera/dswx-s1 > docker/dockerimg_dswx_s1_$tag.tar

# remove image
docker image rm opera/dswx-s1:$tag    
