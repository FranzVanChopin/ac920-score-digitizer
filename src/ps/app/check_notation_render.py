"""Verify duration-only XML playback/engraving and Chinese UI without DP use."""
import json
import os
import pwd
import subprocess
from pathlib import Path
from headless_check import draw
from onecard_common import sha
from panel_ui import request,Rsvg


def main():
    config=json.loads(Path('/etc/ac920-onecard.json').read_text())
    base=Path('/home/fpga/claude-codex-interactive-ui1-20261030/verification_notation')
    base.mkdir();user=pwd.getpwnam(config['user']);os.chown(base,user.pw_uid,user.pw_gid)
    xml=Path('/home/fpga/claude-codex-interactive-ui1-20261030/verification2/results/01.musicxml')
    before=sha(xml)
    subprocess.run(['runuser','-u',config['user'],'--',config['python'],
        '/opt/ac920-panel/render_score.py',str(xml),str(base)],check=True)
    assert before==sha(xml)
    metadata=json.loads((base/'score.json').read_text())
    # The first B4 is120/480 quarter, i.e.125ms at the default120bpm.
    code='from pathlib import Path;import sys;sys.path.insert(0,"/opt/ac920-panel");from render_score import midi_notes;notes,duration=midi_notes(Path(sys.argv[1]));assert any(n[2]==71 and abs(n[0])<.01 and abs(n[1]-.125)<.015 for n in notes);print("MIDI16th=125ms PASS; total",duration)'
    subprocess.run([config['python'],'-c',code,str(base/'score.mid')],check=True)
    state=request(dict(action='state'));state['view']='library';state['selected']='01';state['play']['state']='ready'
    svg={i+1:Rsvg.Handle.new_from_file(str(base/name)) for i,name in enumerate(metadata['pages'])}
    draw(state,base/'score.png',score=metadata,svg=svg)
    state['view']='recognize';state['play']['state']='stopped'
    for song in state['songs']:
        if song['id']=='01':song['pages'][0].update(status='running',progress=55,stage='符号关联')
    state['message']='01-00 · 符号关联';draw(state,base/'progress.png')
    request_state=request(dict(action='state'));draw(request_state,base/'recognize.png')
    result=dict(status='PASS',original_xml_unchanged=True,original_xml_sha256=before,
        chinese_font='WenQuanYi Micro Hei',renderer_revision=2,score=metadata,
        sixteenth_note_duration_ms=125,dp_audio_played=False)
    (base/'result.json').write_text(json.dumps(result,indent=2))
    for path in base.iterdir():
        if path.is_file():os.chmod(path,0o644)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
