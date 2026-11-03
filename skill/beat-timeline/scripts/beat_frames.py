"""Calculate frame anchors from exact rational beat times; no media playback/editing."""
import argparse,json
from fractions import Fraction

def timeline(bpm,fps,beats,start=0):
    bpm,fps=Fraction(str(bpm)),Fraction(str(fps))
    if bpm<=0 or fps<=0 or not 1<=beats<=10000:raise ValueError('Positive BPM/FPS and 1..10000 beats required')
    rows=[]
    for beat in range(beats+1):
        exact=Fraction(start)+beat*60*fps/bpm
        frame=(exact+Fraction(1,2)).numerator//(exact+Fraction(1,2)).denominator
        rows.append(dict(beat=beat,frame=frame,seconds=str(Fraction(frame,1)/fps),rounding_error_frames=str(Fraction(frame)-exact)))
    return dict(bpm=str(bpm),fps=str(fps),anchors=rows,last_frame=rows[-1]['frame'],method='Each anchor rounded from its exact absolute time, without accumulating a rounded beat step')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bpm',required=True);p.add_argument('--fps',required=True);p.add_argument('--beats',required=True,type=int);p.add_argument('--start-frame',type=int,default=0);a=p.parse_args()
    print(json.dumps(timeline(a.bpm,a.fps,a.beats,a.start_frame),ensure_ascii=False,indent=2))
