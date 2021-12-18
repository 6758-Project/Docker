#!/bin/bash

echo "Build docker command"
docker build --tag app_server -f Dockerfile.serving . --rm=False
