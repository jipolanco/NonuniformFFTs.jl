#!/usr/bin/env bash

cd build/1 || exit 1
python3 -m http.server --bind localhost
