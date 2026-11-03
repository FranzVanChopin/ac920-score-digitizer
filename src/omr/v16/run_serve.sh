#!/bin/sh
# Resident v16 on the AC920 (pipelined: the next page's decode, Homr preprocessing and network pass run while
# the current page finishes; same output as single runs): imports, models and C++ libraries are loaded once, then one page per stdin line
# "PHOTO<TAB>NEW_OUT_DIR[<TAB>METER]" (absolute paths); one JSON report per page on stdout, {"ready": true} first.
# usage:  sh run_serve.sh [CODEX_APP_DIR] < pages.tsv
#         printf '/Image/00-00.png\t/home/fpga/claude-out1\n' | sh run_serve.sh
# Same flags and outputs as run_board.sh (the first page of a run is identical to a single run).
set -e
APP=${1:-/home/fpga/omr-transfer-20261024/sdcard_omr_ac920_20261024/app}
PY=${PY:-/home/fpga/ac920-omr-runtime/venv/bin/python}
cd "$(dirname "$0")"
PKG=$(pwd)
[ -f cdec/libdecoder.so ] || sh cdec/build_cdec.sh >&2
[ -f hacc/libhomrgeo.so ] || sh hacc/build_hacc.sh >&2
[ -f racc/librelgeo.so ] || sh racc/build_racc.sh >&2
[ -f sacc/libpilresize.so ] || sh sacc/build_pilresize.sh >&2
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/home/fpga/claude-pycache}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec "$PY" run_page.py --serve --pipeline --bundle "$APP" --threads 4 \
    --csolve --fast-homr --fast-relation --fast-redecode \
    --shared "$PKG/shared" --relation-models "$PKG/shared/relation_models" --link-margin 0.5
