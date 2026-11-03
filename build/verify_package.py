"""Offline, read-only integrity and Python syntax check. No board or audio access."""
from pathlib import Path
import ast, hashlib, json
root = Path(__file__).resolve().parents[1]
OMR = root/'src'/'omr'/'v16'          # v16 package (moved from recognizer/v16 in the 2026-11-02 restructure)
m = json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
python_count = 0
for entry in m['files']:
    p = root/entry['path']
    raw = p.read_bytes()
    assert len(raw) == entry['bytes'], entry['path']
    assert hashlib.sha256(raw).hexdigest() == entry['sha256'], entry['path']
    if p.suffix == '.py':
        ast.parse(raw, filename=str(p)); python_count += 1
up = json.loads((OMR/'MANIFEST.json').read_text(encoding='utf-8'))
for name, e in up['files'].items():
    raw = (OMR/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == e['sha256'], name
print(f"PASS: {len(m['files'])} files intact; {python_count} Python sources parsed; v16 {len(up['files'])}/{len(up['files'])} original assets verified.")
print('Identity: DP_STABLE720_RESPONSE1 / a4_interactive_ui1_light2 / 1280x720@60')
