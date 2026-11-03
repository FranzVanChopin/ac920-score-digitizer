#!/bin/sh
# Build libpilresize.so next to this script (Linux / aarch64 board); see pilresize.cpp.
cd "$(dirname "$0")" && g++ -O3 -ffp-contract=off -fPIC -shared -std=c++17 pilresize.cpp -o libpilresize.so && echo built libpilresize.so
