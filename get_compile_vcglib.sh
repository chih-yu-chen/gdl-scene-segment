#!/bin/bash
docker build -t qmake .
docker run -d -it --rm --name qmake --mount type=bind,source="$(pwd)/app",target=/app qmake

