"""Dense SVG + shared measure timeline + sampled piano at four pitch-preserving rates."""
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from notation_view import prepare

REVISION = 3
RATES = (.75, 1., 1.25, 1.5)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def midi_notes(midi):
    import mido
    notes, active, at = [], {}, 0.
    for message in mido.MidiFile(midi):
        at += message.time
        if message.type not in ('note_on', 'note_off'): continue
        key = (message.channel, message.note)
        if message.type == 'note_on' and message.velocity:
            active.setdefault(key, []).append((at, message.velocity))
        elif active.get(key):
            start, velocity = active[key].pop(0)
            notes.append((start, max(.001, at-start), message.note, velocity))
    for (_, note), entries in active.items():
        for start, velocity in entries: notes.append((start, max(.001, at-start), note, velocity))
    return notes, max(at, max((x[0]+x[1] for x in notes), default=0))

def dynamics(midi):
    """Voicing only: preserve ticks, note lengths/pitches and source dynamic deviations."""
    import mido
    file = mido.MidiFile(midi)
    events, highest = [], {}
    for track in file.tracks:
        tick = 0
        for message in track:
            tick += message.time
            events.append((tick, message))
            if message.type == 'note_on' and message.velocity and message.channel != 9:
                highest[tick] = max(highest.get(tick, 0), message.note)
    origin, measure, half = 0, file.ticks_per_beat*4, file.ticks_per_beat*2
    velocities = []
    for tick, message in sorted(events, key=lambda value: value[0]):
        if message.type == 'time_signature':
            origin = tick
            measure = file.ticks_per_beat*4*message.numerator/message.denominator
            half = measure/2 if message.numerator in (4, 6, 12) else -1
        if message.type != 'note_on' or not message.velocity or message.channel == 9: continue
        phase = (tick-origin) % measure
        accent = 5 if phase == 0 else 3 if phase == half else 0
        base = 98 if message.note == highest[tick] and message.note >= 60 else 84 if message.note >= 60 else 76
        message.velocity = max(35, min(115, base+accent+message.velocity-90))
        velocities.append(message.velocity)
    file.save(midi)
    return dict(min=min(velocities, default=0), max=max(velocities, default=0),
                values=sorted(set(velocities)), note_ons=len(velocities),
                method='upper melody98 / inner84 / bass76; downbeat+5, secondary+3; source deviations retained')

def rate_midi(source, target, speed):
    import mido
    if speed not in RATES: raise ValueError('unsupported playback speed')
    file = mido.MidiFile(source)
    found = False
    for track in file.tracks:
        for message in track:
            if message.type == 'set_tempo':
                message.tempo = max(1, round(message.tempo/speed)); found = True
    if not found: file.tracks[0].insert(0, mido.MetaMessage('set_tempo', tempo=round(500000/speed)))
    file.save(target)

def soundfont():
    requested = os.environ.get('AC920_PIANO_SF2')
    candidates = [requested] if requested else ['/usr/share/sounds/sf2/FluidR3_GM.sf2', '/usr/share/sounds/sf2/TimGM6mb.sf2']
    for candidate in candidates:
        if candidate and Path(candidate).is_file(): return Path(candidate)
    raise RuntimeError('钢琴 SoundFont 未安装，请安装 FluidR3 GM 音色包')

def piano(midi, font, target):
    executable = shutil.which('fluidsynth')
    if not executable: raise RuntimeError('FluidSynth 未安装，无法生成钢琴音频')
    command = [executable, '-ni', '-r', '48000', '-R', '1', '-C', '0', '-g', '.4',
               '-T', 'wav', '-O', 's16', '-F', str(target), str(font), str(midi)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    target.with_suffix('.log').write_text(result.stdout)
    if result.returncode: raise RuntimeError('FluidSynth 生成失败：'+result.stdout[-600:])
    with wave.open(str(target), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (48000, 2, 2):
            raise RuntimeError('钢琴音频格式不是 48kHz stereo16')
        return audio.getnframes()/audio.getframerate()

def multiply(a, b):
    aa, ab, ac, ad, ae, af = a; ba, bb, bc, bd, be, bf = b
    return (aa*ba+ac*bb, ab*ba+ad*bb, aa*bc+ac*bd, ab*bc+ad*bd, aa*be+ac*bf+ae, ab*be+ad*bf+af)

def transform(text):
    matrix = (1, 0, 0, 1, 0, 0)
    for name, data in re.findall(r'(\w+)\(([^)]*)\)', text):
        value = [float(n) for n in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', data)]
        if name == 'translate': next_matrix = (1, 0, 0, 1, value[0], value[1] if len(value)>1 else 0)
        elif name == 'scale': next_matrix = (value[0], 0, 0, value[-1], 0, 0)
        elif name == 'matrix' and len(value) == 6: next_matrix = tuple(value)
        elif name == 'rotate':
            angle = math.radians(value[0]); c, s = math.cos(angle), math.sin(angle)
            next_matrix = (c, s, -s, c, 0, 0)
            if len(value) == 3:
                next_matrix = multiply(multiply((1,0,0,1,value[1],value[2]), next_matrix), (1,0,0,1,-value[1],-value[2]))
        else: raise ValueError('unsupported SVG transform: '+name)
        matrix = multiply(matrix, next_matrix)
    return matrix

def page_geometry(svg):
    """Project Verovio bounding rects into root viewBox, then strip helper groups."""
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns); ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.fromstring(svg)
    view = [float(x) for x in root.attrib['viewBox'].split()]
    measures = {}
    def visit(node, matrix, owner=None, bbox=False):
        if node is not root and node.tag == '{'+ns+'}svg':
            inner = [float(x) for x in node.attrib['viewBox'].split()]
            scale = min(view[2]/inner[2], view[3]/inner[3])
            matrix = multiply(matrix, (scale,0,0,scale,(view[2]-inner[2]*scale)/2-inner[0]*scale,
                                      (view[3]-inner[3]*scale)/2-inner[1]*scale))
        matrix = multiply(matrix, transform(node.get('transform','')))
        classes = node.get('class','').split()
        bbox = bbox or 'bounding-box' in classes
        if 'measure' in classes and not bbox:
            owner = node.get('id')
            measures[owner] = dict(id=owner, number=node.get('data-n',''), points=[])
        if owner and bbox and node.tag == '{'+ns+'}rect':
            x,y,w,h = [float(node.get(k,'0')) for k in ('x','y','width','height')]
            a,b,c,d,e,f = matrix
            for px,py in ((x,y),(x+w,y),(x,y+h),(x+w,y+h)):
                measures[owner]['points'].append((a*px+c*py+e,b*px+d*py+f))
        for child in node: visit(child, matrix, owner, bbox)
    visit(root, (1,0,0,1,0,0))
    for value in measures.values():
        points = value.pop('points')
        if points:
            xs,ys = zip(*points)
            value['bbox'] = [min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)]
    for parent in root.iter():
        for child in list(parent):
            if 'bounding-box' in child.get('class','').split(): parent.remove(child)
    return ET.tostring(root, encoding='unicode'), view, measures

def timeline(toolkit, mapping, locations):
    measures, turns = [], []
    for value in mapping:
        identifier = value.get('measureOn')
        if identifier and identifier not in locations:
            identifier = toolkit.getNotatedIdForElement(identifier) or identifier
        if identifier:
            location = locations.get(identifier)
            if location: measures.append(dict(ms=value['tstamp'], **location))
        ids = value.get('on', []) + value.get('restsOn', [])
        page = locations.get(identifier, {}).get('page') if identifier else None
        if not page and ids:
            notated = toolkit.getNotatedIdForElement(ids[0]) or ids[0]
            page = toolkit.getPageWithElement(notated)
        if page and page > 0 and (not turns or turns[-1]['page'] != page):
            turns.append(dict(ms=value['tstamp'], page=page))
    return measures, turns

def render(xml, destination):
    import verovio
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    font = soundfont(); fingerprint = sha(xml)
    if (destination/'score.json').is_file():
        cached = json.loads((destination/'score.json').read_text())
        if (cached.get('renderer_revision') == REVISION and cached.get('xml_sha256') == fingerprint
            and cached.get('soundfont',{}).get('path') == str(font)
            and all((destination/rate['wav']).is_file() for rate in cached.get('rates',{}).values())
            and len(cached.get('rates',{})) == 4): return cached
    toolkit = verovio.toolkit()
    options = dict(pageWidth=2600, pageHeight=1170, scale=40, pageMarginTop=25,
        pageMarginBottom=25, pageMarginLeft=30, pageMarginRight=30, adjustPageHeight=False,
        breaks='auto', header='none', footer='none', svgViewBox=True, svgBoundingBoxes=True,
        svgAdditionalAttribute=['measure@n'])
    toolkit.setOptions(options)
    display_xml, normalization = prepare(Path(xml).read_text())
    if not toolkit.loadData(display_xml): raise ValueError('MusicXML 排版失败')
    pages, geometries, locations = [], {}, {}
    for number in range(1, toolkit.getPageCount()+1):
        filename = 'page-%03d.svg' % number
        svg, geometry, boxes = page_geometry(toolkit.renderToSVG(number))
        (destination/filename).write_text(svg, encoding='utf-8')
        pages.append(filename); geometries[str(number)] = geometry
        for key, item in boxes.items(): locations[key] = dict(page=number, **item)
    midi = destination/'score.mid'
    midi.write_bytes(base64.b64decode(toolkit.renderToMIDI()))
    velocity = dynamics(midi)
    mapping = toolkit.renderToTimemap(dict(includeMeasures=True, includeRests=True))
    if isinstance(mapping,str): mapping = json.loads(mapping)
    measures, turns = timeline(toolkit, mapping, locations)
    notes, duration = midi_notes(midi)
    if not notes or duration > 1800: raise ValueError('乐谱没有音符，或超出 30 分钟试听上限')
    if not measures or any(not measure.get('bbox') for measure in measures):
        raise RuntimeError('当前小节时间映射或 SVG 边界缺失')
    rates = {}
    for speed in RATES:
        stem = 'score' if speed == 1 else 'score-%03d' % round(speed*100)
        scaled = destination/(stem+'.mid')
        if speed != 1: rate_midi(midi,scaled,speed)
        tail_duration = piano(scaled,font,destination/(stem+'.wav'))
        rates[format(speed,'g')] = dict(wav=stem+'.wav', duration=duration/speed, wav_duration=tail_duration)
    result = dict(pages=pages, duration=duration, turns=turns, measures=measures,
        page_geometry=geometries, page_layout=options, rates=rates, notes=len(notes),
        velocity=velocity, audio='FluidSynth2.2.5 sampled piano, 48kHz stereo16, reverb ON',
        soundfont=dict(path=str(font),bytes=font.stat().st_size,sha256=sha(font)),
        verovio=toolkit.getVersion(), renderer_revision=REVISION, xml_sha256=fingerprint,
        display_normalization=normalization)
    temporary = destination/'score.json.tmp'
    temporary.write_text(json.dumps(result,ensure_ascii=False)); temporary.replace(destination/'score.json')
    print('[SCORE] rendered %d pages, %d notes, %.1fs, %d timed measures, four piano speeds' %
          (len(pages), len(notes), duration, len(measures)), flush=True)
    return result

if __name__ == '__main__': render(Path(sys.argv[1]),Path(sys.argv[2]))
