"""AC920 ARM64/Linux OMR preflight and SD-card transcription runner.

Only standard-library modules are imported before preflight, so this file can
explain an incomplete installation without importing ONNX Runtime first.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

from align_homr_geometry import align_score_geometry, align_score_geometry_with_gaps
from repair_anchored_timing import repair as repair_anchored_timing
from repair_augmentation_dots import repair as repair_augmentation_dots
from repair_duplicate_noteheads import repair_score_from_noteheads
from repair_ledger_chords import repair as repair_ledger_chords
from repair_octave_marks import repair as repair_octave_marks
from repair_pitch_context import repair as repair_pitch_context
from repair_prefix_symbols import repair as repair_prefix_symbols
from repair_rest_bars import repair as repair_rest_bars
from repair_low_octaves import repair as repair_low_octaves
from repair_tuplets import repair as repair_tuplets
from repair_grace_timing import repair as repair_grace_timing
from repair_homr_tremolo import repair as repair_homr_tremolo
from repair_voice_continuity import repair as repair_voice_continuity
from repair_unison_voices import repair as repair_unison_voices, repair_shared_heads, repair_overlapping_ending
from repair_rest_geometry import repair as repair_rest_geometry
from omr_bar_geometry import barline_review
from repair_voice_timing import repair_score

try:
    import resource
except ImportError:  # lets the Windows packaging host inspect the bundle
    resource = None


MIN_GLIBC = (2, 28)
MIN_AVAILABLE_KIB = 1_500_000
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_bundle(bundle: Path) -> dict:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"unsafe manifest path: {relative}")
            continue
        path = bundle / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            errors.append(f"hash mismatch: {relative}")
    return {"bundle": str(bundle), "files": len(manifest["files"]), "errors": errors}


def available_memory_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def runtime_preflight() -> dict:
    errors = []
    warnings = []
    machine = platform.machine().lower()
    libc_name, libc_version = platform.libc_ver()
    try:
        libc_tuple = tuple(int(part) for part in libc_version.split(".")[:2])
    except ValueError:
        libc_tuple = (0, 0)
    if sys.platform != "linux" or machine not in {"aarch64", "arm64"}:
        errors.append("requires 64-bit ARM Linux (aarch64)")
    if sys.version_info[:2] != (3, 12):
        errors.append("bundle wheels require Python 3.12")
    if libc_name != "glibc" or libc_tuple < MIN_GLIBC:
        errors.append("bundle wheels require glibc >= 2.28")
    available_kib = available_memory_kib()
    if available_kib is not None and available_kib < MIN_AVAILABLE_KIB:
        warnings.append("less than 1.5 GB MemAvailable; inference may exhaust RAM")
    model_status = {}
    homr_spec = importlib.util.find_spec("homr")
    if homr_spec is None or not homr_spec.submodule_search_locations:
        errors.append("homr is not installed in this Python environment")
    else:
        package_dir = Path(next(iter(homr_spec.submodule_search_locations)))
        for folder, stem in (("segmentation", "segnet_"), ("transformer", "encoder_"), ("transformer", "decoder_")):
            matches = list((package_dir / folder).glob(stem + "*.onnx"))
            model_status[stem[:-1]] = [str(x) for x in matches]
            if len(matches) != 1:
                errors.append(f"expected one {stem}*.onnx model; found {len(matches)}")
    providers = []
    if not errors:
        try:
            for module_name in ("cv2", "numpy", "rapidocr", "homr.main"):
                importlib.import_module(module_name)
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CPUExecutionProvider" not in providers:
                errors.append("ONNX Runtime CPUExecutionProvider is unavailable")
        except Exception as exc:  # native shared-library load errors matter here
            errors.append(f"OMR dependency import failed: {exc}")
    return {
        "machine": machine,
        "python": platform.python_version(),
        "libc": f"{libc_name} {libc_version}",
        "mem_available_kib": available_kib,
        "onnx_providers": providers,
        "models": model_status,
        "errors": errors,
        "warnings": warnings,
    }


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def parse_meter(text: str) -> tuple[int, int]:
    try:
        beats_text, beat_type_text = text.split("/", 1)
        beats = sum(int(piece) for piece in beats_text.split("+"))
        beat_type = int(beat_type_text)
    except (ValueError, TypeError) as exc:
        raise ValueError("--initial-meter must look like 2/4 or 6/8") from exc
    if beats <= 0 or beat_type <= 0:
        raise ValueError("--initial-meter must use positive numbers")
    return beats, beat_type


def last_meter(score: Path) -> tuple[int, int]:
    root = ET.parse(score).getroot()
    parts = root.findall("part")
    if root.tag != "score-partwise" or not parts:
        raise ValueError("--previous-score is not a MusicXML score-partwise file")
    meter = None
    for measure in parts[0].findall("measure"):
        for attributes in measure.findall("attributes"):
            time_element = attributes.find("time")
            if time_element is not None:
                beats = time_element.findtext("beats", "")
                beat_type = time_element.findtext("beat-type", "")
                meter = parse_meter(f"{beats}/{beat_type}")
    if meter is None:
        raise ValueError("--previous-score has no time signature to carry forward")
    return meter


def normalize_measure_rests(root: ET.Element,
                            initial_meter: tuple[int, int] | None = None) -> int:
    """Give a lone whole-measure rest the active meter's actual duration.

    Homr can emit a four-quarter whole rest in a 2/4 measure. This is a
    deterministic MusicXML repair: the whole-rest glyph denotes a complete
    measure when that staff has no other notes. Adjacent ``backup`` elements
    are shortened too, preserving the next voice's onset at beat zero.
    """
    repaired = 0
    for part in root.findall("part"):
        divisions = 1
        meter: tuple[int, int] | None = None
        for measure_index, measure in enumerate(part.findall("measure")):
            if measure_index == 0 and initial_meter is not None:
                # Continuation pages may show only the key signature. Add the
                # inherited time signature so this page is independently valid.
                if not any(a.find("time") is not None for a in measure.findall("attributes")):
                    attributes = measure.find("attributes")
                    if attributes is None:
                        attributes = ET.Element("attributes")
                        measure.insert(0, attributes)
                    time_element = ET.SubElement(attributes, "time")
                    ET.SubElement(time_element, "beats").text = str(initial_meter[0])
                    ET.SubElement(time_element, "beat-type").text = str(initial_meter[1])
            for attributes in measure.findall("attributes"):
                value = attributes.findtext("divisions")
                if value is not None:
                    divisions = int(value)
                time = attributes.find("time")
                if time is not None:
                    beats = time.findtext("beats", "")
                    beat_type = time.findtext("beat-type", "")
                    try:
                        meter = (sum(int(item) for item in beats.split("+")), int(beat_type))
                    except ValueError:
                        meter = None
            if meter is None or divisions <= 0 or meter[1] <= 0:
                continue
            if measure.attrib.get("implicit") == "yes":
                continue
            target = Fraction(meter[0] * 4 * divisions, meter[1])
            if target.denominator != 1:
                continue
            by_staff: dict[str, list[ET.Element]] = {}
            for note in measure.findall("note"):
                by_staff.setdefault(note.findtext("staff", "1"), []).append(note)
            for notes in by_staff.values():
                if len(notes) != 1:
                    continue
                note = notes[0]
                rest = note.find('rest')
                if rest is None or (note.findtext('type') != 'whole' and rest.get('measure') != 'yes'):
                    continue
                duration = note.find("duration")
                if duration is None or duration.text is None:
                    continue
                old_duration = int(duration.text)
                if old_duration == target.numerator:
                    continue
                duration.text = str(target.numerator)
                note.find("rest").set("measure", "yes")
                written_type = note.find("type")
                if written_type is not None:
                    note.remove(written_type)
                following = list(measure)
                next_index = following.index(note) + 1
                if next_index < len(following) and following[next_index].tag == "backup":
                    backup = following[next_index].find("duration")
                    if backup is not None and backup.text == str(old_duration):
                        backup.text = str(target.numerator)
                repaired += 1
    return repaired


def audit_timing(root: ET.Element) -> dict:
    """Flag notes placed beyond the active measure length for focused review."""
    overfull = []
    missing_meter = []
    for part_index, part in enumerate(root.findall("part"), 1):
        divisions = 1
        meter: tuple[int, int] | None = None
        for measure_index, measure in enumerate(part.findall("measure"), 1):
            cursor = Fraction(0)
            previous_onset = Fraction(0)
            max_end = Fraction(0)
            for element in measure:
                if element.tag == "attributes":
                    value = element.findtext("divisions")
                    if value is not None:
                        divisions = int(value)
                    time_element = element.find("time")
                    if time_element is not None:
                        meter = parse_meter(f"{time_element.findtext('beats', '')}/"
                                            f"{time_element.findtext('beat-type', '')}")
                elif element.tag in ("backup", "forward"):
                    shift = Fraction(int(element.findtext("duration", "0")), divisions)
                    cursor += shift if element.tag == "forward" else -shift
                elif element.tag == "note":
                    is_chord = element.find("chord") is not None
                    is_grace = element.find("grace") is not None
                    onset = previous_onset if is_chord else cursor
                    if not is_chord:
                        previous_onset = onset
                    if not is_grace:
                        duration = Fraction(int(element.findtext("duration", "0")), divisions)
                        max_end = max(max_end, onset + duration)
                        if not is_chord:
                            cursor += duration
            label = {"part": part_index, "measure": measure_index}
            if meter is None:
                missing_meter.append(label)
            elif max_end > Fraction(4 * meter[0], meter[1]):
                overfull.append({**label, "last_event_end_quarters": str(max_end),
                                 "meter": f"{meter[0]}/{meter[1]}"})
    return {"overfull_measures": overfull, "measures_without_meter": missing_meter}


def reflow_complete_voices(root: ET.Element) -> list[dict]:
    """Repair interleaved voice streams only when every voice fills the bar.

    Homr occasionally writes notes from two complete voices in alternating
    order without sufficient MusicXML backups. This changes their onsets even
    though pitch, duration and voice order are correct. Any incomplete voice,
    forward instruction or mid-measure direction leaves the measure untouched.
    """
    overfull = {(item["part"], item["measure"])
                for item in audit_timing(root)["overfull_measures"]}
    repaired = []
    for part_index, part in enumerate(root.findall("part"), 1):
        divisions = 1
        meter = None
        for measure_index, measure in enumerate(part.findall("measure"), 1):
            for attributes in measure.findall("attributes"):
                if attributes.findtext("divisions") is not None:
                    divisions = int(attributes.findtext("divisions"))
                time_element = attributes.find("time")
                if time_element is not None:
                    meter = parse_meter(f"{time_element.findtext('beats')}/"
                                        f"{time_element.findtext('beat-type')}")
            if (part_index, measure_index) not in overfull or meter is None:
                continue
            target = Fraction(meter[0] * 4 * divisions, meter[1])
            if target.denominator != 1:
                continue
            elements = list(measure)
            note_indices = [i for i, element in enumerate(elements) if element.tag == "note"]
            if not note_indices:
                continue
            first, last = note_indices[0], note_indices[-1]
            if any(element.tag not in {"note", "backup"}
                   for element in elements[first:last + 1]):
                continue
            if any(element.tag != "barline" for element in elements[last + 1:]):
                continue
            voices: OrderedDict[str, list[ET.Element]] = OrderedDict()
            for note in measure.findall("note"):
                voice = note.findtext("voice")
                if voice is None:
                    break
                voices.setdefault(voice, []).append(note)
            else:
                if len(voices) < 2:
                    continue
                durations = [sum(int(note.findtext("duration", "0")) for note in notes
                                 if note.find("chord") is None and note.find("grace") is None)
                             for notes in voices.values()]
                if not all(duration == target.numerator for duration in durations):
                    continue
                rebuilt = elements[:first]
                for voice_index, notes in enumerate(voices.values()):
                    if voice_index:
                        backup = ET.Element("backup")
                        ET.SubElement(backup, "duration").text = str(target.numerator)
                        rebuilt.append(backup)
                    rebuilt.extend(notes)
                rebuilt.extend(elements[last + 1:])
                measure[:] = rebuilt
                repaired.append({"part": part_index, "measure": measure_index})
    return repaired


def same_pitch_slur_review(root: ET.Element) -> list[dict]:
    """Locate curves that may be ties; keep them as slurs until reviewed."""
    candidates = []
    for part_index, part in enumerate(root.findall("part"), 1):
        active: dict[tuple[int, str], tuple[int, int]] = {}
        for measure_index, measure in enumerate(part.findall("measure"), 1):
            for note in measure.findall("note"):
                pitch = note.find("pitch")
                if pitch is None:
                    continue
                step = pitch.findtext("step")
                if step not in PITCH_CLASS:
                    continue
                midi = ((int(pitch.findtext("octave")) + 1) * 12 +
                        PITCH_CLASS[step] + int(pitch.findtext("alter", "0")))
                staff = int(note.findtext("staff", "1"))
                for slur in note.findall("notations/slur"):
                    key = (staff, slur.attrib.get("number", "1"))
                    kind = slur.attrib.get("type")
                    if kind == "stop" and key in active:
                        first_measure, first_midi = active.pop(key)
                        if first_midi == midi:
                            candidates.append({"part": part_index, "staff": staff,
                                               "start_measure": first_measure,
                                               "stop_measure": measure_index,
                                               "midi": midi,
                                               "review": "same-pitch slur may be a tie"})
                    elif kind == "start":
                        active[key] = (measure_index, midi)
    return candidates


def normalize_rest_chords(root: ET.Element) -> list[dict]:
    """Separate a rest and pitched chord tones without changing their events.

    A rest cannot be the leading pitched tone of a chord. Homr occasionally
    emits that structure, which crashes Verovio's MusicXML importer. Keep the
    rest, split the pitches into a separate voice, and preserve their onsets.
    This is serialization repair, not evidence that the rest was recognized.
    """
    changes = []
    for part_index, part in enumerate(root.findall("part"), 1):
        existing = {n.findtext("voice", "1") for n in part.findall("measure/note")}
        next_voice = max((int(v) for v in existing if v.isdigit()), default=0)+1
        assigned = {}
        for index, measure in enumerate(part.findall("measure"), 1):
            elements, rebuilt, i = list(measure), [], 0
            while i < len(elements):
                e = elements[i]
                rebuilt.append(e)
                i += 1
                if (e.tag != "note" or e.find("rest") is None or
                        e.find("chord") is not None or e.find("grace") is not None):
                    continue
                end = i
                while (end < len(elements) and elements[end].tag == "note" and
                       elements[end].find("chord") is not None):
                    end += 1
                group = elements[i:end]
                duration = e.findtext("duration")
                key = (e.findtext("staff", "1"), e.findtext("voice", "1"))
                if (not group or duration is None or int(duration) <= 0 or
                        any(n.find("pitch") is None or n.findtext("duration") != duration or
                            (n.findtext("staff", "1"), n.findtext("voice", "1")) != key
                            for n in group)):
                    continue
                if key not in assigned:
                    assigned[key] = str(next_voice)
                    next_voice += 1
                backup = ET.Element("backup")
                ET.SubElement(backup, "duration").text = duration
                rebuilt.append(backup)
                group[0].remove(group[0].find("chord"))
                for n in group:
                    voice = n.find("voice")
                    if voice is None:
                        voice = ET.Element("voice")
                        children = list(n)
                        location = next((j for j, child in enumerate(children)
                                         if child.tag in {"type", "dot", "accidental", "time-modification",
                                                          "stem", "notehead", "staff", "beam", "notations"}),
                                        len(children))
                        n.insert(location, voice)
                    voice.text = assigned[key]
                rebuilt.extend(group)
                i = end
                changes.append({"part": part_index, "measure": index,
                                "method": "rest_chord_separate_voice",
                                "new_voice": assigned[key], "pitched_notes": len(group)})
            measure[:] = rebuilt
        # A rest can also appear as a chord member after a pitched note.
        # Represent that rest as an independent voice while restoring the
        # original cursor; all pitched and rest onsets remain unchanged.
        for index,measure in enumerate(part.findall('measure'),1):
            rebuilt=[];cursor=previous=0
            for element in list(measure):
                if element.tag in {'backup','forward'}:
                    d=int(element.findtext('duration','0'))
                    cursor+=d if element.tag=='forward' else -d
                elif element.tag=='note':
                    chord=element.find('chord');duration=int(element.findtext('duration','0'))
                    if chord is not None and element.find('rest') is not None and duration>0 and cursor>=previous>=0:
                        old_cursor=cursor
                        if cursor>previous:
                            shift=ET.Element('backup');ET.SubElement(shift,'duration').text=str(cursor-previous);rebuilt.append(shift)
                        element.remove(chord)
                        key=(element.findtext('staff','1'),element.findtext('voice','1'),'chord_rest')
                        if key not in assigned:
                            assigned[key]=str(next_voice);next_voice+=1
                        voice=element.find('voice')
                        if voice is None:
                            voice=ET.Element('voice');element.insert(list(element).index(element.find('duration'))+1,voice)
                        voice.text=assigned[key]
                        rebuilt.append(element)
                        cursor=previous+duration
                        if cursor!=old_cursor:
                            shift=ET.Element('backup' if cursor>old_cursor else 'forward')
                            ET.SubElement(shift,'duration').text=str(abs(cursor-old_cursor));rebuilt.append(shift)
                        cursor=old_cursor
                        changes.append({'part':part_index,'measure':index,'method':'chord_rest_separate_voice','new_voice':voice.text})
                        continue
                    if chord is None:
                        previous=cursor
                        if element.find('grace') is None:cursor+=duration
                rebuilt.append(element)
            measure[:]=rebuilt
    return changes


def postprocess_score(root: ET.Element, geometry: dict, image,
                      initial_meter: tuple[int, int] | None = None) -> dict:
    """Run the same image-only repairs on the board and in host regression."""
    prefix = repair_prefix_symbols(root, geometry, image)
    rest_bars = repair_rest_bars(root, geometry, image, initial_meter)
    base_geometry = copy.deepcopy(geometry)
    for system in base_geometry.get("systems", []):
        for staff in system.get("staffs", []):
            staff["notes"] = [n for n in staff["notes"] if not n.get("extended_ledger")]
    from route_trials.testset3_playback_repair import normalize_measure_rests_safe
    normalizer_report = normalize_measure_rests_safe(root, normalize_measure_rests, initial_meter)
    normalized = normalizer_report['normalized_rests']
    # The transactional normalizer replaces XML nodes, so bind the live notes.
    note_x, alignment = align_score_geometry(root, base_geometry)
    reflowed = reflow_complete_voices(root)
    voices = repair_score(root, initial_meter, audit_timing_fn=audit_timing,
                          parse_meter_fn=parse_meter, note_x=note_x)
    duplicates = repair_score_from_noteheads(root, base_geometry, initial_meter, audit_timing)
    gap_x, gap_alignment = align_score_geometry_with_gaps(root, base_geometry)
    anchored = repair_anchored_timing(root, gap_x, audit_timing, parse_meter)
    ledger = repair_ledger_chords(root, geometry)
    low_octaves = repair_low_octaves(root, geometry, image)
    dots = repair_augmentation_dots(root, geometry, image, initial_meter)
    pitch = repair_pitch_context(root, geometry, image)
    octave = repair_octave_marks(root, geometry, image)
    rest_chords = normalize_rest_chords(root)
    grace = repair_grace_timing(root, geometry, image, initial_meter)
    unison = repair_unison_voices(root, geometry, image, initial_meter)
    unison.extend(repair_shared_heads(root, geometry, image))
    unison.extend(repair_overlapping_ending(root, geometry, image, initial_meter))
    rests = repair_rest_geometry(root, geometry, image)
    tuplets = repair_tuplets(root, geometry, image, initial_meter)
    tremolos = repair_homr_tremolo(root)
    continuity = repair_voice_continuity(root, geometry, image)
    return {"geometry_alignment": alignment, "gap_geometry_alignment": gap_alignment,
            "staff_retry": geometry.get("staff_retry", []),
            "staff_recovery": geometry.get("staff_recovery", []),
            "barline_review": barline_review(root, geometry, image),
            "normalized_measure_rests": normalized, "rest_backup_normalization": normalizer_report,
            "reflowed_measures": reflowed,
            "voice_search_measures": voices, "duplicate_notehead_measures": duplicates,
            "anchored_timing_measures": anchored, "ledger_repairs": ledger,
            "augmentation_dot_repairs": dots, "octave_mark_repairs": octave,
            "prefix_symbol_repairs": prefix,
            "rest_bar_repairs": rest_bars, "low_octave_repairs": low_octaves,
            "tuplet_repairs": tuplets,
            "tremolo_serialization_repairs": tremolos,
            "voice_continuity_repairs": continuity,
            "pitch_context_repairs": pitch, "grace_timing_repairs": grace,
            "unison_voice_repairs": unison,
            "rest_geometry_repairs": rests,
            "rest_chord_serialization_repairs": rest_chords}


def transcribe(image: Path, output: Path, sd_root: Path | None,
               initial_meter: tuple[int, int] | None = None,
               previous_score: Path | None = None,
               experimental_ties: bool = False,
               relation_models: Path | None = None,
               relations: str = 'auto',
               staff_scale_pass: bool = False) -> dict:
    preflight = runtime_preflight()
    if preflight["errors"]:
        raise RuntimeError("preflight failed: " + "; ".join(preflight["errors"]))
    image = image.resolve(strict=True)
    if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("input must be an existing JPEG or PNG")
    output = output.resolve()
    if output.suffix.lower() != ".musicxml":
        raise ValueError("output filename must end in .musicxml")
    if initial_meter is not None and previous_score is not None:
        raise ValueError("use either --initial-meter or --previous-score")
    if previous_score is not None:
        previous_score = previous_score.resolve(strict=True)
        if not previous_score.is_file():
            raise ValueError("--previous-score must be an existing MusicXML file")
    if sd_root is not None:
        sd_root = sd_root.resolve(strict=True)
        if not os.path.ismount(sd_root):
            raise ValueError(f"SD root is not a mounted filesystem: {sd_root}")
        if (not within(image, sd_root) or not within(output, sd_root) or
                (previous_score is not None and not within(previous_score, sd_root))):
            raise ValueError("input, output and previous score must be inside --sd-root")
    if previous_score is not None:
        initial_meter = last_meter(previous_score)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ac920-omr-") as temporary:
        staged = Path(temporary) / ("input" + image.suffix.lower())
        shutil.copyfile(image, staged)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        command = [sys.executable, str(Path(__file__).resolve().with_name("homr_with_geometry.py")),
                   str(staged), "--extend-ledger", "--save-preprocessed", "--staff-retry", "--recover-staff",
                   "--recover-structure"]
        if initial_meter is not None:
            command.extend(["--initial-meter", f"{initial_meter[0]}/{initial_meter[1]}"])
        stage_start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
        stage_times = {'homr_child_s': time.monotonic()-stage_start}
        if result.returncode:
            raise RuntimeError(f"homr exited {result.returncode}: {result.stderr[-4000:]}")
        generated = staged.with_suffix(".musicxml")
        if not generated.is_file():
            raise RuntimeError("homr returned successfully but did not write MusicXML")
        generated_geometry = staged.with_suffix(".geometry.json")
        if not generated_geometry.is_file():
            raise RuntimeError("homr returned successfully but did not write geometry")
        root = ET.parse(generated).getroot()
        parts = root.findall("part")
        measures = sum(len(part.findall("measure")) for part in parts)
        if root.tag != "score-partwise" or not parts or not measures:
            raise RuntimeError("generated file lacks a MusicXML score/part/measure")
        geometry_payload = json.loads(generated_geometry.read_text(encoding="utf-8"))
        import cv2
        preprocessed_path = staged.with_suffix(".preprocessed.png")
        preprocessed = cv2.imread(str(preprocessed_path), cv2.IMREAD_GRAYSCALE)
        if preprocessed is None:
            raise RuntimeError("Homr preprocessed image is missing")
        stage_start = time.monotonic()
        repairs = postprocess_score(root, geometry_payload, preprocessed, initial_meter)
        stage_times['base_postprocess_s'] = time.monotonic()-stage_start
        # Homr's child has exited before loading the small CPU models. Keep the
        # original result whenever inference or an integrity guard fails.
        stage_start = time.monotonic()
        relation_report = {"enabled": False, "reason": "relation models not installed"}
        models = relation_models or Path(__file__).resolve().parent / 'relation_models'
        if relations != 'off' and (models / 'manifest.json').is_file():
            try:
                from board_relation_runtime import enhance
                ET.ElementTree(root).write(generated, encoding='utf8', xml_declaration=True)
                enhanced = staged.with_name('enhanced.musicxml')
                scale_options={'supplementary_staff_scale':16.} if staff_scale_pass else {}
                relation_report = enhance(image, generated_geometry, generated, models, enhanced,write_merge_report=False,**scale_options)
                enhanced_root = ET.parse(enhanced).getroot()
                if (enhanced_root.tag != 'score-partwise' or len(enhanced_root.findall('part')) != 1 or
                        not enhanced_root.findall('part/measure')):
                    raise ValueError('Relation enhancement produced an empty or unsupported score')
                root = enhanced_root
                relation_report['enabled'] = True
            except Exception as exc:
                relation_report = {"enabled": False, "fallback": True, "error": str(exc)}
        elif relations == 'off':
            relation_report = {"enabled": False, "reason": "disabled by option"}
        elif relation_models is not None:
            relation_report = {"enabled": False, "fallback": True, "error": "relation model manifest missing"}
        repairs['relation_merge'] = relation_report
        stage_times['relation_total_s'] = time.monotonic()-stage_start
        stage_start = time.monotonic()
        structure_report={'enabled':False,'reason':'validated relation assets unavailable'}
        if relations!='off' and (models/'manifest.json').is_file():
            from route_trials.board_structure_retry import candidates as structure_candidates, recover as recover_structure
            from board_relation_runtime import enhance as structure_enhance
            proof=structure_candidates(geometry_payload,preprocessed)
            root,geometry_payload,structure_report=recover_structure(
                root,geometry_payload,preprocessed,image,staged,command,env,models,relations,
                staff_scale_pass,initial_meter,postprocess_score,structure_enhance,proof)
        repairs['structure_recovery']=structure_report
        tail_report={'enabled':False,'reason':'primary relation enhancement unavailable'}
        if structure_report.get('retried'):
            tail_report={'enabled':False,'reason':'one alternate recognition already used for local structure'}
        elif relation_report.get('enabled'):
            from route_trials.board_tail_retry_experiment import recover as recover_tail
            from board_relation_runtime import enhance as tail_enhance
            # Run the alternate model only after the primary model has exited;
            # merge only certified new bars, preserving every existing measure.
            root,geometry_payload,tail_report=recover_tail(root,geometry_payload,preprocessed,image,staged,command,env,models,relations,staff_scale_pass,initial_meter,postprocess_score,tail_enhance)
        repairs['tail_splice']=tail_report
        stage_times['structure_and_tail_s'] = time.monotonic()-stage_start
        stage_start = time.monotonic()
        from route_trials.testset3_playback_repair import repair_negative_backups
        repairs['negative_backup_recovery'] = repair_negative_backups(root, geometry_payload)
        from route_trials.testset3_duplet_recovery import repair as recover_duplets
        repairs['classical_duplet_recovery'] = recover_duplets(root, geometry_payload, preprocessed, initial_meter)
        from route_trials.testset3_tie_octave_guard import capture_existing, bind, validate_after_octave
        existing_ties = capture_existing(root)
        from route_trials.testset3_tie_recovery_v6 import recover as recover_ties
        repairs['classical_tie_recovery'] = recover_ties(root, geometry_payload, preprocessed)
        tie_octave_guard = bind(root, repairs['classical_tie_recovery'], existing=existing_ties)
        from route_trials.testset3_octave_recovery_v3 import repair as recover_octaves
        repairs['classical_octave_recovery'] = recover_octaves(root, geometry_payload, preprocessed, previous_score=previous_score)
        repairs['tie_after_octave_review'] = validate_after_octave(root, tie_octave_guard)
        stage_times['classical_final_repairs_s'] = time.monotonic()-stage_start
        repairs['stage_elapsed_s'] = stage_times
        parts = root.findall('part')
        measures = sum(len(part.findall('measure')) for part in parts)
        tie_repair = None
        if experimental_ties:
            from repair_ties import repair as repair_ties
            tie_repair = repair_ties(copy.deepcopy(root), geometry_payload,
                                     preprocessed)
            tie_repair["applied"] = False
            tie_repair["warning"] = (
                "Tie and same-pitch slur arcs remain ambiguous; proposals need review")
        timing_review = audit_timing(root)
        tie_review = same_pitch_slur_review(root)
        ET.ElementTree(root).write(generated, encoding="utf-8", xml_declaration=True)
        geometry_payload["input_sha256"] = sha256(image)
        geometry_payload["score_sha256"] = sha256(generated)
        geometry_payload["alignment"] = repairs["geometry_alignment"]
        from route_trials.relation_trace_sidecar import persist_edge_traces
        persist_edge_traces(repairs,output)
        geometry_payload["repairs"] = repairs
        generated_geometry.write_text(
            json.dumps(geometry_payload, ensure_ascii=False) + "\n",
            encoding="utf-8")
        descriptor, temporary_output = tempfile.mkstemp(prefix=".omr-", suffix=".musicxml", dir=output.parent)
        try:
            with os.fdopen(descriptor, "wb") as sink, generated.open("rb") as source:
                shutil.copyfileobj(source, sink)
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary_output, output)
        finally:
            if os.path.exists(temporary_output):
                os.unlink(temporary_output)
        geometry_output = output.with_suffix(".geometry.json")
        descriptor, temporary_geometry = tempfile.mkstemp(
            prefix=".omr-geometry-", suffix=".json", dir=output.parent)
        try:
            with os.fdopen(descriptor, "wb") as sink, generated_geometry.open("rb") as source:
                shutil.copyfileobj(source, sink)
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary_geometry, geometry_output)
        finally:
            if os.path.exists(temporary_geometry):
                os.unlink(temporary_geometry)
        tie_review_output = None
        if tie_repair is not None:
            tie_review_output = output.with_suffix(".tie_review.json")
            descriptor, temporary_review = tempfile.mkstemp(
                prefix=".omr-tie-review-", suffix=".json", dir=output.parent)
            try:
                payload = {"score_sha256": sha256(output), "applied": False,
                           "proposals": tie_repair}
                with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
                    json.dump(payload, sink, ensure_ascii=False, indent=2)
                    sink.write("\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                os.replace(temporary_review, tie_review_output)
            finally:
                if os.path.exists(temporary_review):
                    os.unlink(temporary_review)
    return {
        "input": str(image),
        "input_sha256": sha256(image),
        "output": str(output),
        "output_sha256": sha256(output),
        "geometry_output": str(geometry_output),
        "geometry_sha256": sha256(geometry_output),
        **repairs,
        "parts": len(parts),
        "measures_total": measures,
        "timing_review": timing_review,
        "tie_review": tie_review,
        "tie_review_output": str(tie_review_output) if tie_review_output else None,
        "experimental_tie_proposals": tie_repair,
        "initial_meter_supplied": (f"{initial_meter[0]}/{initial_meter[1]}"
                                   if initial_meter is not None else None),
        "previous_score": str(previous_score) if previous_score is not None else None,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if resource else None,
        "warning": "MusicXML is structurally readable; optical correctness still needs review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-bundle")
    inspect.add_argument("bundle", type=Path)
    commands.add_parser("preflight")
    run = commands.add_parser("transcribe")
    run.add_argument("image", type=Path)
    run.add_argument("output", type=Path)
    run.add_argument("--sd-root", type=Path, help="require both paths on this mounted SD filesystem")
    run.add_argument("--initial-meter", type=parse_meter,
                     help="carry a preceding page's time signature, e.g. 2/4")
    run.add_argument("--previous-score", type=Path,
                     help="read the active time signature from the preceding MusicXML page")
    run.add_argument("--tie-review-geometry", "--experimental-ties",
                     dest="experimental_ties", action="store_true",
                     help="save geometry-based tie proposals without changing MusicXML")
    run.add_argument('--relations', choices=('auto','off'), default='auto',
                     help='Use installed CPU relation models with guarded fallback')
    run.add_argument('--relation-models', type=Path,
                     help='Integrity-checked ONNX relation model folder')
    run.add_argument('--staff-scale-pass', action='store_true',
                     help='Add sequential staff-normalized symbol inference; retain the full photograph')
    args = parser.parse_args()
    try:
        if args.command == "inspect-bundle":
            result = check_bundle(args.bundle)
        elif args.command == "preflight":
            result = runtime_preflight()
        else:
            result = transcribe(args.image, args.output, args.sd_root,
                                args.initial_meter, args.previous_score,
                                args.experimental_ties, args.relation_models, args.relations,args.staff_scale_pass)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("errors") else 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
