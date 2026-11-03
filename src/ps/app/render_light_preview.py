"""Render light-theme visual states offline; never starts playback or sends keys."""
import copy
import json
import sys
from pathlib import Path
sys.path.append('/opt/ac920-panel')
from headless_check import draw
from panel_ui import request,Rsvg

folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
actual=request(dict(action='state'),timeout=2)
draw(actual,folder/'recognize.png')
progress=copy.deepcopy(actual);progress['view']='recognize';progress['selected']='01';progress['busy']=True
for song in progress['songs']:
    if song['id']=='01':song['pages'][0].update(status='running',progress=55,stage='符号关联')
progress['message']='01-00 · 符号关联'
draw(progress,folder/'progress.png')
library=copy.deepcopy(actual);library['view']='library';library['selected']='01'
library['message']='长按 S2 打开当前曲目并开始播放'
draw(library,folder/'library_ready.png')
score_dir=Path('/home/fpga/claude-codex-interactive-ui1-20261030/verification_notation')
metadata=json.loads((score_dir/'score.json').read_text())
svg={i+1:Rsvg.Handle.new_from_file(str(score_dir/name)) for i,name in enumerate(metadata['pages'])}
library['play']['state']='ready'
draw(library,folder/'score.png',score=metadata,svg=svg)
empty=copy.deepcopy(actual);empty['songs']=[];empty['selected']=None
draw(empty,folder/'empty.png')
draw(actual,folder/'error.png',audio_error='DP 音频设备未就绪，请检查连接。')
(folder/'states.json').write_text(json.dumps(dict(theme='light1',basis='live catalogue; progress/error are layout fixtures',
    views=['recognize','progress','library_ready','score','empty','error'],no_audio_played=True),indent=2))
print('PASS light offscreen states:recognize,progress,library,score,empty,error; no DP/audio used.')
