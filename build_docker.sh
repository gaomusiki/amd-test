#!/bin/bash

docker_prefix=docker.io/library
docker_name="a3_env"
docker_version=0

# build docker image
docker build -t "$docker_name:v$docker_version" .

# save docker image tar
docker save -o "${docker_name}_v${docker_version}".tar "$docker_prefix/$docker_name:v$docker_version"