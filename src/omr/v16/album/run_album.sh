#!/bin/sh
# Album mode on the AC920: recognize what the capture firmware (Codex a4_album1) wrote under ALBUM_ROOT
# (Grey/SS-PP.bmp + Meta/SS-PP.json), one resident pipelined recognizer for all pages, and write one MusicXML per
# song (OUT_DIR/SS.musicxml).  Run it again after shooting more pages: finished pages are skipped.
# usage, from any directory:
#   sh album/run_album.sh ALBUM_ROOT NEW_OR_EARLIER_OUT_DIR [--rotate ccw|cw|180|none|auto] [--watch SECONDS] [--songs 0,1]
# Environment: APP (Codex app dir), PY (the board runtime's venv python), WORK (upright copies; default /dev/shm).
set -e
ALBUM=$1
OUT=$2
if [ -z "$ALBUM" ] || [ -z "$OUT" ]; then
    echo "usage: sh album/run_album.sh ALBUM_ROOT OUT_DIR [--rotate ccw] [--watch SECONDS] [--songs 0,1]"
    exit 2
fi
shift 2
APP=${APP:-/home/fpga/omr-transfer-20261024/sdcard_omr_ac920_20261024/app}
PY=${PY:-/home/fpga/ac920-omr-runtime/venv/bin/python}
WORK=${WORK:-/dev/shm/omr-album-$(id -u)}
ALBUM=$(readlink -f "$ALBUM")
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT")
cd "$(dirname "$0")/.."
PKG=$(pwd)
[ -f cdec/libdecoder.so ] || sh cdec/build_cdec.sh >&2
[ -f hacc/libhomrgeo.so ] || sh hacc/build_hacc.sh >&2
[ -f racc/librelgeo.so ] || sh racc/build_racc.sh >&2
[ -f sacc/libpilresize.so ] || sh sacc/build_pilresize.sh >&2
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/home/fpga/claude-pycache}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec "$PY" album/omr_album.py --album "$ALBUM" --out "$OUT" --work "$WORK" "$@" -- \
    "$PY" run_page.py --serve --pipeline --bundle "$APP" --threads 4 \
    --csolve --fast-homr --fast-relation --fast-redecode \
    --shared "$PKG/shared" --relation-models "$PKG/shared/relation_models" --link-margin 0.5
