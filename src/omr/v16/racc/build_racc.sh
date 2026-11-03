#!/bin/sh
# Build librelgeo.so next to this script (Linux / aarch64 board).  -ffp-contract=off: no fused
# multiply-add, every operation rounded as in CPython; no -ffast-math.
cd "$(dirname "$0")" && g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 relgeo.cpp -o librelgeo.so && echo built librelgeo.so
