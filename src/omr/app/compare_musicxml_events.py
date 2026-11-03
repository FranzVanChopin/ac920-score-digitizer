"""Compare the musical events in two aligned piano MusicXML scores.

This is a diagnostic for generated-photo experiments, not a measure of
performance on unseen camera photographs. All timing uses exact quarter-note
fractions, so MusicXML files with different ``divisions`` remain comparable.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path


PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def read_measures(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    parts = root.findall("part")
    if root.tag != "score-partwise" or len(parts) != 1:
        raise ValueError(f"Expected one unnamespaced score-partwise piano part: {path}")
    divisions = 1
    result = []
    for measure in parts[0].findall("measure"):
        cursor = Fraction(0)
        previous_onset = Fraction(0)
        notes: Counter[tuple] = Counter()
        rests: Counter[tuple] = Counter()
        ties: Counter[tuple] = Counter()
        grace = 0
        invisible = 0
        for element in measure:
            if element.tag == "attributes":
                value = element.findtext("divisions")
                if value is not None:
                    divisions = int(value)
                    if divisions <= 0:
                        raise ValueError("MusicXML divisions must be positive")
            elif element.tag in ("backup", "forward"):
                step = Fraction(int(element.findtext("duration", "0")), divisions)
                cursor += step if element.tag == "forward" else -step
            elif element.tag == "note":
                is_chord = element.find("chord") is not None
                is_grace = element.find("grace") is not None
                onset = previous_onset if is_chord else cursor
                if not is_chord:
                    previous_onset = onset
                duration = Fraction(int(element.findtext("duration", "0")), divisions)
                if not is_chord and not is_grace:
                    cursor += duration
                # Hidden playback notes do not appear in the photographed
                # engraving. They still advance the MusicXML cursor.
                if element.get("print-object") == "no":
                    invisible += 1
                    continue
                if is_grace:
                    grace += 1
                    continue
                staff = int(element.findtext("staff", "1"))
                pitch = element.find("pitch")
                if pitch is not None:
                    step = pitch.findtext("step")
                    if step not in PITCH_CLASS:
                        raise ValueError(f"Unknown pitch step: {step}")
                    midi = (int(pitch.findtext("octave")) + 1) * 12
                    midi += PITCH_CLASS[step] + int(pitch.findtext("alter", "0"))
                    notes[(staff, onset, midi, duration)] += 1
                    for tie in element.findall("tie"):
                        ties[(staff, onset, midi, tie.attrib.get("type", ""))] += 1
                elif element.find("rest") is not None:
                    rests[(staff, onset, duration)] += 1
        result.append({"notes": notes, "rests": rests, "ties": ties, "grace": grace,
                       "invisible_excluded": invisible})
    return result


def project(notes: Counter[tuple], kind: str) -> Counter[tuple]:
    projected: Counter[tuple] = Counter()
    if kind == "pitch_duration":
        for (staff, _, midi, duration), count in notes.items():
            projected[(staff, midi, duration)] += count
    elif kind == "pitch":
        for (staff, _, midi, _), count in notes.items():
            projected[(staff, midi)] += count
    else:
        raise ValueError(kind)
    return projected


def overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())


def printable(items: Counter, limit: int = 5) -> list[dict]:
    return [{"event": [str(value) for value in key], "count": count}
            for key, count in sorted(items.items(), key=lambda entry: str(entry[0]))[:limit]]


def compare(reference: list[dict], candidate: list[dict], start: int) -> dict:
    if start < 1 or start + len(candidate) - 1 > len(reference):
        raise ValueError("Candidate measures are outside the reference score")
    rows = []
    for local_number, predicted in enumerate(candidate, 1):
        source_number = start + local_number - 1
        expected = reference[source_number - 1]
        note_overlap = overlap(expected["notes"], predicted["notes"])
        pitch_duration_overlap = overlap(project(expected["notes"], "pitch_duration"),
                                         project(predicted["notes"], "pitch_duration"))
        pitch_overlap = overlap(project(expected["notes"], "pitch"),
                                project(predicted["notes"], "pitch"))
        rest_overlap = overlap(expected["rests"], predicted["rests"])
        tie_overlap = overlap(expected["ties"], predicted["ties"])
        missing_notes = expected["notes"] - predicted["notes"]
        extra_notes = predicted["notes"] - expected["notes"]
        row = {
            "reference_measure": source_number,
            "candidate_measure": local_number,
            "reference_notes": sum(expected["notes"].values()),
            "candidate_notes": sum(predicted["notes"].values()),
            "exact_note_events": note_overlap,
            "same_pitch_duration_in_measure": pitch_duration_overlap,
            "same_pitch_in_measure": pitch_overlap,
            "reference_rests": sum(expected["rests"].values()),
            "candidate_rests": sum(predicted["rests"].values()),
            "exact_rest_events": rest_overlap,
            "reference_ties": sum(expected["ties"].values()),
            "candidate_ties": sum(predicted["ties"].values()),
            "exact_ties": tie_overlap,
            "reference_grace": expected["grace"],
            "candidate_grace": predicted["grace"],
            "reference_invisible_excluded": expected.get("invisible_excluded", 0),
            "candidate_invisible_excluded": predicted.get("invisible_excluded", 0),
        }
        if missing_notes or extra_notes:
            row["missing_note_examples"] = printable(missing_notes)
            row["extra_note_examples"] = printable(extra_notes)
        if expected["rests"] != predicted["rests"]:
            row["missing_rest_examples"] = printable(expected["rests"]-predicted["rests"])
            row["extra_rest_examples"] = printable(predicted["rests"]-expected["rests"])
        rows.append(row)
    summed = {name: sum(row[name] for row in rows)
              for name in ("reference_notes", "candidate_notes", "exact_note_events",
                           "same_pitch_duration_in_measure", "same_pitch_in_measure",
                           "reference_rests", "candidate_rests", "exact_rest_events",
                           "reference_ties", "candidate_ties", "exact_ties",
                           "reference_grace", "candidate_grace",
                           "reference_invisible_excluded", "candidate_invisible_excluded")}
    summed["exact_note_measures"] = sum(row["reference_notes"] == row["candidate_notes"] ==
                                        row["exact_note_events"] for row in rows)
    summed["measure_count"] = len(rows)
    summed["note_event_precision"] = (summed["exact_note_events"] / summed["candidate_notes"]
                                      if summed["candidate_notes"] else None)
    summed["note_event_recall"] = (summed["exact_note_events"] / summed["reference_notes"]
                                   if summed["reference_notes"] else None)
    return {"start_reference_measure": start, "totals": summed, "measures": rows,
            "limitations": [
                "Exact note event means same measure, staff, onset, MIDI pitch and performed duration.",
                "Notes marked print-object=no are excluded, while their timing is retained.",
                "This does not score engraving, dynamics, articulation, fingering, slurs, lyrics or repeat playback.",
                "This generated-photo benchmark uses source material already seen during development; it is not holdout accuracy.",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--start-measure", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(read_measures(args.reference), read_measures(args.candidate),
                     args.start_measure)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
