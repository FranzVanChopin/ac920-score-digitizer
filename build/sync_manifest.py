"""Rebuild PACKAGE_MANIFEST.json from the repository's tracked file set.

The repository carries a per-file SHA-256 manifest; build/verify_package.py fails as soon
as any tracked file changes. Run this after editing, adding or renaming a file:

    python build/sync_manifest.py --check        # read-only, exit 1 if stale
    python build/sync_manifest.py --dry-run      # read-only, list the changes
    python build/sync_manifest.py                # rewrite the manifest

Which files count:
  * inside a git work tree  -> exactly `git ls-files` (what a clone will contain), so
    .git/ and untracked local junk are never listed;
  * otherwise (no .git yet) -> a disk walk that skips .git/ and __pycache__/.

The manifest never lists itself. Existing `origin` notes are preserved for paths that are
still present; new paths get a dated note so provenance stays visible.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
SELF_NAME = "PACKAGE_MANIFEST.json"
SKIP_DIRS = {".git", "__pycache__", ".Xil", "xsim.dir", ".venv", "venv", "node_modules"}
REQUIRED = ("README.md", "LICENSE", "PACKAGE_MANIFEST.json")


def tracked_paths() -> tuple[list[str], str]:
    """Return (repo-relative paths, how they were obtained)."""
    if (ROOT / ".git").exists():
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "-z"],
                capture_output=True, check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"error: git ls-files failed: {exc}", file=sys.stderr)
            raise SystemExit(2)
        names = [p for p in proc.stdout.decode("utf-8", "surrogateescape").split("\0") if p]
        if not names:
            print("error: this is a git repo but nothing is staged/tracked.\n"
                  "       run `git add -A` first, then re-run this script.", file=sys.stderr)
            raise SystemExit(2)
        missing = [r for r in REQUIRED if r not in names]
        if missing:
            print(f"error: tracked set looks incomplete, missing {missing}.\n"
                  "       run `git add -A` first.", file=sys.stderr)
            raise SystemExit(3)
        return names, "git ls-files"

    names = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file():
            names.append(path.relative_to(ROOT).as_posix())
    return names, "disk walk (no .git yet)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild PACKAGE_MANIFEST.json.")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 when the manifest is out of date")
    args = ap.parse_args()

    if not MANIFEST.is_file():
        print(f"error: {MANIFEST} not found", file=sys.stderr)
        return 2

    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old = {e["path"]: e for e in spec.get("files", [])}
    names, source = tracked_paths()
    today = datetime.date.today().isoformat()

    entries, added, gone_missing, changed = [], [], [], []
    present = set()
    for rel in sorted(names):
        if rel == SELF_NAME:
            continue
        path = ROOT / rel
        if not path.is_file():
            gone_missing.append(rel)
            continue
        present.add(rel)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        prev = old.get(rel)
        if prev is None:
            added.append(rel)
            origin = f"added {today}: {rel}"
        else:
            origin = prev.get("origin", "")
            if prev.get("sha256") != digest or prev.get("bytes") != len(data):
                changed.append(rel)
        entries.append({
            "path": rel,
            "bytes": len(data),
            "sha256": digest,
            "mode": prev.get("mode", "0o644") if prev else "0o644",
            "origin": origin,
        })

    dropped = sorted(set(old) - present)
    stale = bool(added or dropped or changed or gone_missing)

    print(f"root    : {ROOT}")
    print(f"source  : {source}")
    print(f"entries : {len(old)} -> {len(entries)}")
    for label, items in (("added", added), ("dropped", dropped),
                         ("changed", changed), ("vanished", gone_missing)):
        print(f"{label:9s}: {len(items)}")
        for item in items:
            print(f"    {item}")

    if args.check or args.dry_run:
        print("\nmanifest is STALE" if stale else "\nmanifest is up to date")
        return 1 if stale else 0

    spec["files"] = entries
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\nwritten: {MANIFEST}")
    print("next   : python build/verify_package.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
