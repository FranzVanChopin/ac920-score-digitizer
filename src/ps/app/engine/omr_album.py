"""Recognize the album the IMX477 capture firmware (Codex a4_album1) writes, with the resident pipeline
(run_page.py --serve --pipeline), and join each song's pages into one MusicXML.

Album (the capture card's FAT32 partition, or any copy of it):
  Grey/SS-PP.bmp    8-bit grey 4056x2868, fixed A4 crop, SENSOR orientation (the page lies on its side)
  Meta/SS-PP.json   written last; "complete": true marks the page as finished
  Image/SS-PP.bmp   colour (not used)
  SS = song 00-99, PP = page 00-99 (a long press of S1 starts the next song).

Output (OUT, never inside the album):
  SS.musicxml       the song: its recognized pages joined (merge_pages.py), rewritten after every page
  pages/SS-PP_<sha8>_<rotation>/  run_page.py's output for that page (page.musicxml, report.json, ...)
  album.json        state: per page the Grey file's sha256, rotation used, share of systems with a clef found,
                    status, measures, notes, seconds
  service.log       the recognizer's own log lines
Pages already recognized (same Grey sha256 and rotation) are skipped, so the command can simply be run again; a page that
failed gets one more try per run.  --page-timeout: a page without a result restarts the recognizer.
A page after the first of its song gets the meter of the page before (run_page METER '@prev' or that page's own).

  omr_album.py --album ROOT --out OUT [--rotate auto|ccw|cw|180|none] [--work DIR] [--watch SECONDS] -- SERVE_CMD...
  SERVE_CMD: the run_page.py --serve --pipeline command line (run_album.sh gives the board's)
--rotate: how to turn the Grey image upright.  ccw (default, the fixed convention from 10-30): the page top lies at
  the RIGHT edge of the sensor image (the capture preview marks it); cw: at its left edge.  A page whose systems
  mostly have no clef found (upside down: its clefs are at the right) gets "orientation_suspect": true in album.json
  and a warning, so the UI can ask for the page to be shot again the right way round.  auto: such a page is
  recognized again turned by 180 degrees, the attempt with more clefs is kept and its orientation used from then on.
--watch: keep running and pick up new pages every SECONDS (capture and recognition on one system); Ctrl+C ends.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_pages  # noqa: E402

NAME = re.compile(r"^(\d{2})-(\d{2})\.json$")
WINDOW = 2            # pages in the recognizer at once
ROTATE = {"ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180, "none": None}
OPPOSITE = {"ccw": "cw", "cw": "ccw", "none": "180", "180": "none"}
CLEFS_OK = 0.5        # share of systems with a clef found above which a page counts as the right way up


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan(album):
    """finished pages: [(song, page, grey_path, meta)] in song / page order"""
    found = []
    meta_dir = album / "Meta"
    if not meta_dir.is_dir():
        return found
    for p in sorted(meta_dir.iterdir()):
        m = NAME.match(p.name)
        if not m:
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):                 # still being written: next scan
            continue
        if meta.get("complete") is not True:
            continue
        grey = album / meta.get("gray_file", f"Grey/{p.stem}.bmp")
        if grey.is_file():
            found.append((int(m.group(1)), int(m.group(2)), grey, meta))
    return found


def lines_horizontal(img):
    """True when the dark structure is mostly horizontal lines (staff lines), i.e. the page is upright or upside
    down rather than on its side: compare the row and the column profile's line-to-line contrast"""
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    dark = 255.0 - small.astype(np.float64)
    rows, cols = dark.mean(axis=1), dark.mean(axis=0)
    return float(np.abs(np.diff(rows)).mean()) > float(np.abs(np.diff(cols)).mean())


def upright(grey, work, rotate):
    img = cv2.imread(str(grey), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {grey}")
    if ROTATE[rotate] is not None:
        img = cv2.rotate(img, ROTATE[rotate])
    dst = work / (grey.stem + ".bmp")
    tmp = work / (grey.stem + ".tmp.bmp")
    cv2.imwrite(str(tmp), img)
    tmp.replace(dst)
    return dst, lines_horizontal(img)


def clef_ratio(run):
    """share of the page's systems where the header stage found a clef (None: no headers.json).  An upside-down
    page gives 0: its clefs sit at the right end of the staves, where nobody looks for them."""
    try:
        h = json.loads((Path(run) / "headers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return round(sum(1 for s in h if any(s.get("clefs", {}).values())) / len(h), 3) if h else 0.0


def page_meter(xml_path):
    try:
        sig = re.findall(r"<beats>(\d+)</beats>\s*<beat-type>(\d+)</beat-type>", Path(xml_path).read_text(encoding="utf-8"))
    except OSError:
        return ""
    return "/".join(sig[-1]) if sig else ""


class Service:
    """run_page.py --serve: one page per stdin line, one JSON line per page on stdout"""

    def __init__(self, cmd, log_path, env=None):
        self.log = open(log_path, "a", encoding="utf-8")
        # own session on Linux: kill() can end the recognizer together with its forked decode workers
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1,
                                  creationflags=0x4000 if os.name == "nt" else 0, start_new_session=os.name != "nt")
        self.results, self.cv = {}, threading.Condition()
        self.ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.log.write(line)
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "ui_stage" in r:
                print("[UI_STAGE]" + json.dumps(r), flush=True)
                continue
            if r.get("ready"):
                self.ready.set()
            elif "photo" in r:
                with self.cv:
                    self.results[r["photo"]] = r
                    self.cv.notify_all()
        self.ready.set()
        with self.cv:
            self.cv.notify_all()

    def send(self, photo, out, meter):
        self.p.stdin.write(f"{photo}\t{out}\t{meter}\n")
        self.p.stdin.flush()

    def take(self, photo, timeout):
        """the report of this photo (None: still running after `timeout` s); raises if the service died"""
        with self.cv:
            if photo not in self.results:
                self.cv.wait(timeout)
            if photo in self.results:
                return self.results.pop(photo)
        if self.p.poll() is not None:
            raise RuntimeError(f"recognizer exited with code {self.p.returncode} (see service.log)")
        return None

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self.kill()
            return
        self.log.close()

    def kill(self):
        """a hung recognizer: end it and its workers"""
        try:
            if os.name == "nt":
                self.p.kill()
            else:
                import signal
                os.killpg(self.p.pid, signal.SIGKILL)
        except OSError:
            pass
        self.p.wait()
        self.log.close()


def main(argv=None, env=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: omr_album.py --album ROOT --out OUT [options] -- SERVE_CMD...")
    k = argv.index("--")
    serve_cmd = argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rotate", choices=sorted(ROTATE) + ["auto"], default="ccw")
    ap.add_argument("--work", type=Path, default=None, help="where the upright copies go (default OUT/work; /dev/shm on the board)")
    ap.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    ap.add_argument("--songs", default=None, help="only these songs, e.g. 0,1")
    ap.add_argument("--page-timeout", type=float, default=600.0,
                    help="seconds without a result for the page in work before the recognizer is restarted and the page "
                         "marked failed (a board page takes ~25 s)")
    a = ap.parse_args(argv[:k])
    album, out = a.album.resolve(), a.out.resolve()
    if out == album or album in out.parents:
        raise SystemExit("--out must not be inside the album (the capture card stays read-only here)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pages").mkdir(exist_ok=True)
    work = (a.work or out / "work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    songs = {int(s) for s in a.songs.split(",")} if a.songs else None
    state_path = out / "album.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"pages": {}}
    old_rotate = state.get("rotate")     # what pages from before rotate_used was recorded were turned with
    state.update(album=str(album), rotate=a.rotate)

    def key(song, page):
        return f"{song:02d}-{page:02d}"

    def rebuild(song):
        done = sorted((v["page"], v["xml"]) for v in state["pages"].values() if v["song"] == song and v["status"] == "ok")
        if not done:
            return
        tree, counts = merge_pages.merge([x for _, x in done], title=f"{song:02d}")
        merge_pages.write(tree, out / f"{song:02d}.musicxml")
        state.setdefault("songs", {})[f"{song:02d}"] = dict(pages=[p for p, _ in done], measures=sum(counts),
                                                             measures_per_page=counts)

    def settled(kk, st):
        """this page's stored result stands (for the same Grey file)"""
        if st["status"] == "failed":
            return kk in tried
        if st["status"] != "ok":
            return False
        used = st.setdefault("rotate_used", old_rotate)
        if a.rotate != "auto":
            return used == a.rotate
        if "clef_ratio" not in st:
            st["clef_ratio"] = clef_ratio(st["run"])
        return used is not None and ((st["clef_ratio"] or 0) >= CLEFS_OK or bool(st.get("orientation_uncertain")))

    svc = None
    backlog = []          # found, not sent yet: (song, page, grey, sha256, size, mtime, rotation or None = current)
    pending = []          # sent, in order: (song, page, upright photo, run dir, time sent, rotation)
    queued = set()        # keys in backlog or pending
    tried = set()         # keys sent in this run: a page that failed earlier gets one more try per run
    last_sent = None      # (song, page) of the last line sent
    last_done = 0.0       # when the recognizer last finished a page (or started)
    try:
        while True:
            for song, page, grey, meta in scan(album):
                kk = key(song, page)
                if (songs is not None and song not in songs) or kk in queued:
                    continue
                st = state["pages"].get(kk)
                size, mtime = grey.stat().st_size, grey.stat().st_mtime
                final = st is not None and settled(kk, st)
                if final and st.get("size") == size and st.get("mtime") == mtime:
                    continue
                digest = sha256(grey)
                if final and st.get("sha256") == digest:
                    st.update(size=size, mtime=mtime)
                    continue
                backlog.append((song, page, grey, digest, size, mtime, None))
                queued.add(kk)
            if backlog and svc is None:
                svc = Service(serve_cmd, out / "service.log", env)
                svc.ready.wait()
                if svc.p.poll() is not None:
                    raise RuntimeError("recognizer did not start (see service.log)")
                log("recognizer ready")
                last_done = time.time()
            # at most WINDOW pages in the recognizer: the next one is known early (pipeline) and the upright
            # copies (11.6 MB each, in RAM when --work is on /dev/shm) do not pile up
            while backlog and len(pending) < WINDOW:
                song, page, grey, digest, size, mtime, rot = backlog.pop(0)
                kk = key(song, page)
                retry = rot is not None
                if rot is None:
                    rot = a.rotate if a.rotate != "auto" else state.get("rotate_auto", "ccw")
                photo, horizontal = upright(grey, work, rot)
                if a.rotate == "auto" and not horizontal and not retry:   # an upright photo, not one on its side
                    rot = "none" if rot in ("ccw", "cw") else "ccw"
                    photo, horizontal = upright(grey, work, rot)
                run = out / "pages" / f"{kk}_{digest[:8]}_{rot}"
                if run.exists():                      # an interrupted earlier attempt: keep it aside
                    run.rename(run.with_name(f"{run.name}.old{int(time.time())}"))
                prev = state["pages"].get(key(song, page - 1)) if page > 0 else None
                if page > 0 and last_sent == (song, page - 1):
                    meter = "@prev"
                elif prev and prev["status"] == "ok":
                    meter = page_meter(prev["xml"])
                else:
                    meter = ""
                attempts = state["pages"][kk].get("attempts", {}) if retry else {}
                state["pages"][kk] = dict(song=song, page=page, grey=str(grey), sha256=digest, size=size, mtime=mtime,
                                          status="running", run=str(run), meter_in=meter, lines_horizontal=horizontal,
                                          rotate_used=rot, attempts=attempts)
                if not horizontal:
                    log(f"{kk}: staff lines look vertical after rotation {rot}; check the camera orientation")
                svc.send(photo, run, meter)
                tried.add(kk)
                pending.append((song, page, photo, run, time.time(), rot))
                last_sent = (song, page)
                log(f"{kk}: sent ({grey.name}, rotation {rot}, meter {meter or 'auto'})")
            save_json(state_path, state)
            if pending:
                song, page, photo, run, t0, rot = pending[0]
                left = a.page_timeout - (time.time() - max(t0, last_done))
                r = svc.take(str(photo), timeout=max(0.2, min(1.0 if a.watch is not None else 60.0, left)))
                kk = key(song, page)
                if r is None:
                    if time.time() - max(t0, last_done) > a.page_timeout:
                        log(f"{kk}: no result after {a.page_timeout:.0f} s: restarting the recognizer, page marked failed")
                        svc.kill()
                        svc, last_sent = None, None
                        state["pages"][kk].update(status="failed", error="timeout")
                        queued.discard(kk)
                        photo.unlink(missing_ok=True)
                        # the other pages already sent go back to the front of the backlog
                        again = [state["pages"][key(s2, p2)] for s2, p2, _, _, _, _ in pending[1:]]
                        backlog[:0] = [(v["song"], v["page"], Path(v["grey"]), v["sha256"], v["size"], v["mtime"],
                                        v["rotate_used"]) for v in again]
                        pending.clear()
                        save_json(state_path, state)
                    continue                          # still running (watch: look for new pages meanwhile)
                pending.pop(0)
                last_done = time.time()
                queued.discard(kk)
                st = state["pages"][kk]
                if "error" in r:
                    st.update(status="failed", error=r["error"])
                    log(f"{kk}: FAILED {r['error'][:200]}")
                else:
                    xml = run / "page.musicxml"
                    text = xml.read_text(encoding="utf-8")
                    ratio = clef_ratio(run)
                    result = dict(run=str(run), xml=str(xml), measures=r.get("measures"), seconds=r.get("total_s"),
                                  notes=len(re.findall(r"<note>(?:(?!</note>).)*?<pitch>", text, re.S)),
                                  meter=page_meter(xml), clef_ratio=ratio)
                    st["attempts"][rot] = result
                    upside = ratio is not None and ratio < CLEFS_OK
                    st.update(status="ok", rotate_used=rot, since_sent_s=round(time.time() - t0, 1),
                              orientation_suspect=upside, **result)
                    if a.rotate == "auto" and upside and OPPOSITE[rot] not in st["attempts"]:
                        log(f"{kk}: clefs found on {ratio:.0%} of the systems with rotation {rot}: upside down? "
                            f"trying {OPPOSITE[rot]}")
                        st["status"] = "running"
                        backlog.insert(0, (song, page, Path(st["grey"]), st["sha256"], st["size"], st["mtime"],
                                           OPPOSITE[rot]))
                        queued.add(kk)
                        photo.unlink(missing_ok=True)
                        save_json(state_path, state)
                        continue
                    if a.rotate == "auto" and upside:   # both ways tried: keep the one with more clefs
                        best_rot, best = max(st["attempts"].items(), key=lambda kv: kv[1]["clef_ratio"] or 0)
                        st.update(rotate_used=best_rot, orientation_uncertain=True,
                                  orientation_suspect=(best["clef_ratio"] or 0) < CLEFS_OK, **best)
                        log(f"{kk}: few clefs found either way up; kept rotation {best_rot} (orientation uncertain)")
                    elif upside:
                        log(f"{kk}: clefs found on only {ratio:.0%} of the systems: the page may be upside down "
                            f"(try --rotate {OPPOSITE[rot]} or auto)")
                    if a.rotate == "auto" and not upside:
                        if state.get("rotate_auto") != rot:
                            log(f"orientation for the following pages: {rot}")
                        state["rotate_auto"] = rot
                    rebuild(song)
                    log(f"{kk}: {st['measures']} measures, {st['notes']} notes, clefs on {(st['clef_ratio'] or 0):.0%} "
                        f"of the systems, {st['seconds']} s, rotation {st['rotate_used']} -> {song:02d}.musicxml")
                try:
                    photo.unlink()
                except OSError:
                    pass
                save_json(state_path, state)
                continue
            if a.watch is None:
                break
            time.sleep(a.watch)
    except KeyboardInterrupt:
        log("stopped")
    finally:
        if svc is not None:
            svc.close()
        for v in state["pages"].values():
            if v["status"] == "running":
                v["status"] = "interrupted"
        save_json(state_path, state)
    done = [v for v in state["pages"].values() if v["status"] == "ok"]
    log(f"{len(done)} page(s) recognized; songs: {', '.join(sorted(state.get('songs', {}))) or '-'}")


if __name__ == "__main__":
    main()
