#!/bin/sh
# Build libdecoder.so next to this script (Linux / aarch64 board).  -ffp-contract=off: no fused
# multiply-add, so every product and sum is rounded as in numpy; no -ffast-math (keeps the order).
cd "$(dirname "$0")" && g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 decoder.cpp -o libdecoder.so && echo built libdecoder.so
