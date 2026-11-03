"""Add progress messages to copies of the installed engine. No inference changes."""
import ast
import hashlib
import json
from pathlib import Path


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('Unsupported installed recognizer; progress patch anchor: ' + before[:60])
    return text.replace(before, after)


def make(recognizer, destination):
    recognizer, destination = Path(recognizer), Path(destination)
    original = (recognizer / 'run_page.py').read_text()
    page = replace_once(original, 'HERE = Path(__file__).resolve().parent',
                        'HERE = Path(' + repr(str(recognizer)) + ')')
    anchor = '        if a.profile:\n'
    page = replace_once(page, anchor,
        '        if cur.get("photo") is not None:\n'
        '            print(json.dumps(dict(ui_stage=name, photo=str(cur["photo"]))), flush=True)\n' + anchor)
    album_source = (recognizer / 'album/omr_album.py').read_text()
    album = replace_once(album_source, '            if r.get("ready"):',
        '            if "ui_stage" in r:\n'
        '                print("[UI_STAGE]" + json.dumps(r), flush=True)\n'
        '                continue\n'
        '            if r.get("ready"):')
    for text in (page, album):
        ast.parse(text)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'run_page_ui.py').write_text(page)
    (destination / 'omr_album.py').write_text(album)
    helpers = {}
    for name in ('merge_pages.py', 'engrave.py'):
        source = recognizer / 'album' / name
        if name == 'engrave.py' and not source.is_file():
            continue # Older recognizers did not yet supply the engraving helper.
        raw = source.read_bytes()
        ast.parse(raw)
        (destination / name).write_bytes(raw)
        helpers[name] = hashlib.sha256(raw).hexdigest()
    provenance = dict(original_page_sha256=hashlib.sha256(original.encode()).hexdigest(),
                      original_album_sha256=hashlib.sha256(album_source.encode()).hexdigest(),
                      helper_sha256=helpers,
                      change='Only HERE path and structured progress output; album helpers copied unchanged')
    (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return provenance
