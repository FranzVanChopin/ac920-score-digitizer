"""Small cache checks shared by the resident controller and engraving worker."""
import json,wave
from pathlib import Path

REVISION=3
RATES=(.75,1.,1.25,1.5)

def load_score(directory,fingerprint,font=None):
    directory=Path(directory)
    try:
        value=json.loads((directory/'score.json').read_text(encoding='utf-8'))
        if value.get('renderer_revision')!=REVISION or value.get('xml_sha256','').lower()!=fingerprint.lower():return None
        pages=value.get('pages')
        if not isinstance(pages,list) or not pages or any(not isinstance(p,str) or Path(p).name!=p or not (directory/p).is_file() for p in pages):return None
        if not (directory/'score.mid').is_file():return None
        if font is not None:
            font=Path(font)
            if value.get('soundfont',{}).get('path')!=str(font) or value['soundfont'].get('bytes')!=font.stat().st_size:return None
        return value
    except (OSError,ValueError,TypeError,KeyError):return None

def rate_ready(directory,value,speed):
    if not value:return False
    rate=value.get('rates',{}).get(format(speed,'g'))
    if not isinstance(rate,dict):return False
    name=rate.get('wav')
    if not isinstance(name,str) or Path(name).name!=name:return False
    try:
        with wave.open(str(Path(directory)/name),'rb') as audio:
            return (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())==(48000,2,2) and audio.getnframes()>0
    except (OSError,EOFError,wave.Error):return False
