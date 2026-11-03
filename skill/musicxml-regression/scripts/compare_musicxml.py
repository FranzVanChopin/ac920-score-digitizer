"""Compare partwise MusicXML event multisets; does not score recognition accuracy."""
import argparse,collections,json
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET

def events(path):
    root=ET.fromstring(Path(path).read_bytes())
    for e in root.iter():e.tag=e.tag.split('}')[-1]
    if root.tag!='score-partwise':raise ValueError('Only uncompressed score-partwise MusicXML is supported')
    result=collections.Counter()
    for part in root.findall('part'):
        divisions=Fraction(1)
        for index,measure in enumerate(part.findall('measure')):
            cursor=Fraction(0);last_start=None
            for e in measure:
                if e.tag=='attributes' and e.findtext('divisions'):
                    divisions=Fraction(e.findtext('divisions'))
                    if divisions<=0:raise ValueError('Nonpositive divisions')
                elif e.tag in ('backup','forward'):
                    d=Fraction(e.findtext('duration','0'))/divisions
                    cursor+=d if e.tag=='forward' else -d
                    if cursor<0:raise ValueError('Negative measure cursor')
                    last_start=None
                elif e.tag=='note':
                    grace=e.find('grace') is not None;chord=e.find('chord') is not None
                    duration=Fraction(e.findtext('duration','0'))/divisions
                    if duration<0:raise ValueError('Negative duration')
                    start=last_start if chord else cursor
                    if start is None:raise ValueError('Chord lacks a preceding note')
                    rest=e.find('rest') is not None
                    pitch=e.find('pitch')
                    if rest:value=('rest',)
                    elif pitch is not None:value=('pitch',pitch.findtext('step'),str(Fraction(pitch.findtext('alter','0'))),pitch.findtext('octave'))
                    elif e.find('unpitched') is not None:
                        u=e.find('unpitched');value=('unpitched',u.findtext('display-step'),u.findtext('display-octave'))
                    else:raise ValueError('Unsupported note without pitch/rest/unpitched')
                    ties=tuple(sorted(set(t.get('type') for t in e.findall('tie'))))
                    event=(part.get('id'),index,e.findtext('voice','1'),e.findtext('staff','1'),value,str(start),str(duration),grace,e.find('cue') is not None,ties)
                    result[event]+=1
                    if not chord:
                        last_start=start
                        if not grace:cursor+=duration
    return result

def compare(before,after):
    a,b=events(before),events(after);lost=a-b;added=b-a
    return dict(semantic_events_equal=not lost and not added,baseline_events=sum(a.values()),candidate_events=sum(b.values()),lost_events=sum(lost.values()),added_events=sum(added.values()),lost_examples=[{'event':k,'count':v} for k,v in list(lost.items())[:10]],added_examples=[{'event':k,'count':v} for k,v in list(added.items())[:10]],scope='Exact written-event comparison, not ground-truth accuracy. Does not expand repeats or compare layout/articulations.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');a=p.parse_args()
    try:r=compare(a.before,a.after)
    except (OSError,ValueError,ET.ParseError) as e:print(json.dumps({'valid':False,'error':str(e)}));raise SystemExit(2)
    print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['semantic_events_equal'] else 1)
