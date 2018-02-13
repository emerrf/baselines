#!/bin/bash
set -x

docker build . -t emerrf/baselines-gwt:1.0.0
docker push emerrf/baselines-gwt:1.0.0