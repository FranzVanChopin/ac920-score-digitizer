"""Isolated real one-song inference/render test. Never plays audio or reboots."""
import json
import os
import pwd
import subprocess
import time
from pathlib import Path
import interactive_service as engine
from headless_check import draw
from panel_model import Panel,catalogue
from panel_ui import request,Rsvg


def main():
    config=json.loads(Path('/etc/ac920-onecard.json').read_text())
    destination=Path('/home/fpga/claude-codex-interactive-ui1-20261030/verification2')
    destination.mkdir() # never overwrite a previous verification
    user=pwd.getpwnam(config['user'])
    os.chown(destination,user.pw_uid,user.pw_gid)
    for name in ('results','work','score'):
        path=destination/name;path.mkdir();os.chown(path,user.pw_uid,user.pw_gid)
    initial=request(dict(action='state'))
    assert not initial['busy'] and not initial['return_pending']
    controller=engine.Controller(config)
    controller.panel=Panel(catalogue(controller.mount,destination/'results'))
    controller.panel.selected=1
    song=controller.panel.current()
    assert song['id']=='01'
    started=time.monotonic();events=[]
    command=engine.recognition_command(config,controller.mount,destination/'results',destination/'work','01')
    with (destination/'inference.log').open('w') as log:
        process=subprocess.Popen(command,cwd=config['recognizer'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();controller.progress(song,line)
            if line.startswith('[UI_STAGE]'):
                value=json.loads(line[len('[UI_STAGE]'):]);events.append(value)
                print('[TEST] stage',value['ui_stage'],flush=True)
                if value['ui_stage']=='relation':draw(controller.snapshot(),destination/'progress.png')
        if process.wait():raise RuntimeError('inference failed; see inference.log')
    state=json.loads((destination/'results/album.json').read_text())
    assert set(state['pages'])=={'01-00'}
    assert state['pages']['01-00']['status']=='ok'
    assert {event['ui_stage'] for event in events}>=set(engine.STAGES)
    score=destination/'results/01.musicxml'
    subprocess.run(['runuser','-u',config['user'],'--',config['python'],
        '/opt/ac920-panel/render_score.py',str(score),str(destination/'score')],check=True)
    info=json.loads((destination/'score/score.json').read_text())
    display=controller.snapshot();display['view']='library';display['play']['state']='ready'
    svg={i+1:Rsvg.Handle.new_from_file(str(destination/'score'/name)) for i,name in enumerate(info['pages'])}
    draw(display,destination/'score.png',score=info,svg=svg)
    # Exercise the actual resident controller's selection/cached recognition path.
    value=request(dict(action='key',key='S2',kind='short'));assert value['selected']=='01'
    request(dict(action='key',key='S2',kind='long'))
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        try: final=request(dict(action='state'),timeout=2)
        except OSError: time.sleep(.2);continue
        if not final['busy']:break
        time.sleep(.2)
    assert not final['busy'] and '已完成' in final['message'] and not final['return_pending']
    value=request(dict(action='key',key='S1',kind='long'));assert value['view']=='library'
    value=request(dict(action='key',key='S2',kind='short'));assert value['selected']=='01'
    value=request(dict(action='key',key='S1',kind='long'));assert value['view']=='recognize'
    draw(value,destination/'recognize.png')
    result=dict(status='PASS',song='01',pages=['01-00'],stages=[x['ui_stage'] for x in events],
        seconds=round(time.monotonic()-started,2),score_pages=len(info['pages']),
        score_notes=info['notes'],duration=info['duration'],
        automatic_recognition=False,automatic_return=False,
        gtk_offscreen=True,audio_played=False,physical_dp_test='deferred by user',
        resident_controller_selection_and_cached_recognition='PASS')
    (destination/'result.json').write_text(json.dumps(result,indent=2))
    for path in destination.rglob('*'):
        if path.is_file():os.chmod(path,0o644)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
