"""Compare a two-part treble/bass OMR export against one piano MusicXML part."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from compare_musicxml_events import compare, read_measures


def read_two_parts(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    parts = root.findall("part")
    if len(parts) != 2:
        raise ValueError("Expected separate treble and bass MusicXML parts")
    clefs = [part.findtext("measure/attributes/clef/sign") for part in parts]
    if clefs != ["G", "F"]:
        raise ValueError(f"Expected treble part followed by bass part; got {clefs}")
    parsed = []
    with tempfile.TemporaryDirectory(prefix="omr-two-part-", dir=path.parent) as temporary:
        for index, part in enumerate(parts):
            single = ET.Element("score-partwise")
            single.append(copy.deepcopy(part))
            filename = Path(temporary) / f"part{index + 1}.musicxml"
            ET.ElementTree(single).write(filename, encoding="utf-8",
                                         xml_declaration=True)
            parsed.append(read_measures(filename))
    if len(parsed[0]) != len(parsed[1]):
        raise ValueError("Treble and bass parts have different measure counts")
    merged = []
    for index in range(len(parsed[0])):
        measure = {"notes": Counter(), "rests": Counter(), "ties": Counter(),
                   "grace": 0}
        for staff, part_measures in enumerate(parsed, 1):
            source = part_measures[index]
            for kind in ("notes", "rests", "ties"):
                for event, count in source[kind].items():
                    measure[kind][(staff, *event[1:])] += count
            measure["grace"] += source["grace"]
        merged.append(measure)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--start-measure", type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(read_measures(args.reference), read_two_parts(args.candidate),
                     args.start_measure)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
