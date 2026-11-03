#!/bin/sh
# One page on the AC920 (photo -> MusicXML) with v16: the route A shared network sh8s (shared/, as v15) and every exact
# speed-up (v16: Homr staff geometry in C++ as well).  usage, from any directory:  sh run_board.sh PHOTO NEW_OUT_DIR [CODEX_APP_DIR]
# The four C++ libraries are compiled here on the first call (g++ -O3 -ffp-contract=off).
# Environment: PY (default: the board runtime's venv python), PYTHONPYCACHEPREFIX.
set -e
PHOTO=$1
OUT=$2
APP=${3:-/home/fpga/omr-transfer-20261024/sdcard_omr_ac920_20261024/app}
PY=${PY:-/home/fpga/ac920-omr-runtime/venv/bin/python}
if [ -z "$PHOTO" ] || [ -z "$OUT" ]; then
    echo "usage: sh run_board.sh PHOTO NEW_OUT_DIR [CODEX_APP_DIR]"
    exit 2
fi
if [ -e "$OUT" ]; then
    echo "$OUT exists: give a new directory"
    exit 2
fi
PHOTO=$(readlink -f "$PHOTO")
OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
cd "$(dirname "$0")"
PKG=$(pwd)
[ -f cdec/libdecoder.so ] || sh cdec/build_cdec.sh
[ -f hacc/libhomrgeo.so ] || sh hacc/build_hacc.sh
[ -f racc/librelgeo.so ] || sh racc/build_racc.sh
[ -f sacc/libpilresize.so ] || sh sacc/build_pilresize.sh
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/home/fpga/claude-pycache}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$PY" run_page.py --photo "$PHOTO" --out "$OUT" --bundle "$APP" --threads 4 \
    --csolve --fast-homr --fast-relation --fast-redecode \
    --shared "$PKG/shared" --relation-models "$PKG/shared/relation_models" --link-margin 0.5
sha256sum "$OUT/page.musicxml"
