"""Headless on-board verification; no key commands, camera capture or audible output."""
import copy,json,os,pwd,subprocess,sys,time,wave
from pathlib import Path
sys.path.append('/opt/ac920-panel')
from onecard_common import sha,write_json
from panel_ui import request,Rsvg
from headless_check import draw

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def main():
    config=json.loads(Path('/etc/ac920-onecard.json').read_text())
    folder=ROOT/'verification';folder.mkdir(exist_ok=True)
    user=pwd.getpwnam(config['user']);os.chown(folder,user.pw_uid,user.pw_gid)
    xml=ROOT/'sample.musicxml';before=sha(xml)
    started=time.monotonic()
    subprocess.run(['runuser','-u',config['user'],'--','env',
        'AC920_PIANO_SF2=/usr/share/sounds/sf2/FluidR3_GM.sf2',config['python'],
        str(HERE/'render_score.py'),str(xml),str(folder)],check=True)
    render_s=time.monotonic()-started
    metadata=json.loads((folder/'score.json').read_text())
    assert sha(xml)==before and metadata['renderer_revision']==3
    assert metadata['measures'] and all(m.get('bbox') for m in metadata['measures'])
    audio={}
    for speed,item in metadata['rates'].items():
        with wave.open(str(folder/item['wav']),'rb') as stream:
            assert (stream.getframerate(),stream.getnchannels(),stream.getsampwidth())==(48000,2,2)
            duration=stream.getnframes()/48000
            import array
            peak=0;clipped=0
            while block:=stream.readframes(48000):
                samples=array.array('h',block)
                peak=max(peak,max(map(abs,samples),default=0));clipped+=sum(abs(n)>=32760 for n in samples)
        assert 0<peak<32760 and clipped==0
        assert abs(item['duration']-metadata['duration']/float(speed))<.001
        audio[speed]=dict(peak=peak,clipped=clipped,wav_duration=duration)
    state=request({'action':'state'},timeout=2)
    draw(state,folder/'recognize.png')
    warning=copy.deepcopy(state);warning['view']='recognize';warning['busy']=False
    song=next(s for s in warning['songs'] if s['pages']);warning['selected']=song['id']
    song['pages'][0].update(status='ok',progress=100,stage='已完成',orientation_suspect=True,clef_ratio=0.,rotate_used='ccw')
    draw(warning,folder/'orientation_warning.png')
    library=copy.deepcopy(state);library['view']='library';library['play'].update(state='ready',song='01',speed=1.25)
    svg={i+1:Rsvg.Handle.new_from_file(str(folder/name)) for i,name in enumerate(metadata['pages'])}
    library['speed']=1.25
    for index in (0,min(3,len(metadata['measures'])-1)):
        measure=metadata['measures'][index]
        draw(library,folder/('playback_%d.png'%index),score=metadata,svg=svg,page=measure['page'],
            started=time.monotonic()-(measure['ms']+100)/1000/1.25,speed=1.25)
    result=dict(status='PASS',xml_sha256=before,original_xml_unchanged=True,render_seconds=render_s,
        score=metadata,audio=audio,physical_dp_test=False,audible_output_test=False,
        fixtures='saved MusicXML used for rendering only; orientation warning is injected UI fixture')
    write_json(folder/'result.json',result)
    for p in folder.iterdir():
        if p.is_file():os.chmod(p,0o644)
    print('PASS real FluidSynth four-rate WAVs without clipping; Verovio measure maps; offscreen UI; render_s=%.2f'%render_s)

if __name__=='__main__':main()
