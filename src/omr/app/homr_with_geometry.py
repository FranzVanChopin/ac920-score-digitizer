"""Run Homr once and retain its detected notehead coordinates.

The MusicXML path follows Homr 0.7.0's process_image. The sidecar feeds
conservative geometry-aware repairs in board_homr.py. Optional sparse-staff
recovery uses the existing segmentation; bounded bass retries reuse the
already loaded Transformer sequentially.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.staff_parsing import parse_staffs
from homr.transformer.configs import Config


def describe_geometry(staffs: list, image_shape: tuple[int, ...]) -> dict:
    result = {"preprocessed_size": [int(image_shape[1]), int(image_shape[0])],
              "systems": []}
    for system in staffs:
        entry = {"staffs": []}
        for staff in system.staffs:
            entry["staffs"].append({
                "y_range": [float(staff.min_y), float(staff.max_y)],
                "x_range": [float(staff.min_x), float(staff.max_x)],
                "grid": [{"x": round(float(p.x), 2),
                          "y": [round(float(y), 2) for y in p.y]}
                         for p in staff.grid],
                "notes": [{"x": round(float(note.center[0]), 2),
                           "y": round(float(note.center[1]), 2),
                           "position": int(note.position),
                           "extended_ledger": bool(getattr(note, "_omr_extended_ledger", False)),
                           "staff_space": getattr(note, "_omr_staff_space", None),
                           "stem": ([round(float(value), 2) for value in
                                     (*note.stem.center, *note.stem.size)]
                                    if note.stem is not None else None),
                           "size": [round(float(note.box.size[0]), 2),
                                    round(float(note.box.size[1]), 2)]}
                          for note in staff.get_notes()],
            })
        result["systems"].append(entry)
    return result


def run(image_path: Path, save_preprocessed: bool = False,
        extend_ledger: bool = False, staff_retry: bool = False,
        initial_meter: tuple[int, int] | None = None,
        recover_staff: bool = False, extend_staff_tail: bool = False,
        recover_structure: bool = False) -> tuple[Path, Path]:
    from homr_image_limits import validate_homr_image
    validate_homr_image(image_path)
    recovery = []
    from homr_single_staff import install_single_staff_normalization
    single_staff_records = install_single_staff_normalization()
    if recover_staff:
        from homr_staff_recovery import install_staff_recovery
        recovery = install_staff_recovery()
    if recover_structure:
        from homr_staff_bidirectional import install_staff_extent
        structure_extensions = install_staff_extent(apply_extensions=True)
        tail_candidates = []
    else:
        from homr_staff_extent import install_staff_extent
        tail_candidates = install_staff_extent(apply_extensions=extend_staff_tail)
        structure_extensions = []
    if extend_ledger:
        from homr_ledger_geometry import install_extended_ledger_detection
        install_extended_ledger_detection()
    config = ProcessingConfig(False, False, False, False, -1,
                              False, False, False)
    debug = None
    try:
        rest_only=False
        try:
            staffs, image, debug, title_future = detect_staffs_in_image(
                str(image_path), config)
        except Exception as exc:
            if str(exc) != 'No noteheads found':
                raise
            from homr_rest_only import detect_rest_only
            staffs,image,debug,title_future=detect_rest_only(str(image_path),config)
            rest_only=True
        geometry = describe_geometry(staffs, image.shape)
        geometry["staff_tail_candidates"] = tail_candidates
        geometry["staff_tail_applied"] = extend_staff_tail
        geometry["staff_structure_extensions"] = structure_extensions
        geometry["staff_recovery"] = recovery
        geometry['rest_only_fallback'] = rest_only
        if save_preprocessed:
            saved = image_path.with_suffix(".preprocessed.png")
            if not cv2.imwrite(str(saved), image):
                raise RuntimeError("could not save Homr's preprocessed image")
        transformer_config = Config()
        transformer_config.use_gpu_inference = False
        transformer_config.use_coreml_encoder = False
        symbols = parse_staffs(debug, staffs, image, transformer_config,
                               selected_staff=-1)
        if rest_only and (not symbols or any(s.rhythm.startswith('note_') for voice in symbols for s in voice)
                          or not any(s.rhythm.startswith('rest_') for voice in symbols for s in voice)):
            raise RuntimeError('rest-only decoder produced notes or no rests; abstaining')
        geometry['single_staff_normalization'] = single_staff_records
        title = title_future.result(60)
        score = generate_xml(XmlGeneratorArguments(), symbols, title)
        xml_path = image_path.with_suffix(".musicxml")
        geometry_path = image_path.with_suffix(".geometry.json")
        score.write(str(xml_path))
        if staff_retry:
            from homr_staff_retry import retry_dense_bass
            root = ET.parse(xml_path).getroot()
            geometry["staff_retry"] = retry_dense_bass(
                root, geometry, image, transformer_config, initial_meter)
            ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
        geometry_path.write_text(json.dumps(geometry, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        return xml_path, geometry_path
    finally:
        if debug is not None:
            debug.clean_debug_files_from_previous_runs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.add_argument("--extend-ledger", action="store_true")
    parser.add_argument("--staff-retry", action="store_true")
    parser.add_argument("--recover-staff", action="store_true")
    parser.add_argument("--extend-staff-tail", action="store_true")
    parser.add_argument("--recover-structure", action="store_true",
                        help="Alternate hypothesis with raster-certified staff continuation; callers must fuse only locally verified structure")
    parser.add_argument("--initial-meter", help="meter carried from the preceding page, e.g. 4/4")
    args = parser.parse_args()
    meter = tuple(map(int, args.initial_meter.split("/"))) if args.initial_meter else None
    if meter is not None and (len(meter) != 2 or min(meter) <= 0):
        parser.error("initial meter must have two positive integers")
    xml_path, geometry_path = run(args.image, args.save_preprocessed, args.extend_ledger,
                                  args.staff_retry, meter, args.recover_staff, args.extend_staff_tail,
                                  args.recover_structure)
    print(json.dumps({"score": str(xml_path), "geometry": str(geometry_path)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
