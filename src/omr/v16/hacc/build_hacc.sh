#!/bin/sh
# Build libhomrgeo.so next to this script (Linux / aarch64 board).  -ffp-contract=off: no fused
# multiply-add, so every product and sum is rounded as in CPython; no -ffast-math.
cd "$(dirname "$0")" && g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 homrgeo.cpp -o libhomrgeo.so && echo built libhomrgeo.so
