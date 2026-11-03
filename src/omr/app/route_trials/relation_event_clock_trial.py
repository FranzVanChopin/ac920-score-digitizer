"""Bounded photographic reconstruction of complete existing event clocks.

The decoded durations, pitches, labels and chord membership are retained.
An unknown meter may be replaced by agreement between two complete clocks
inside the bar; no global time signature is introduced. Reference scores are
never inputs to this module.
"""
import copy
from collections import Counter, defaultdict
from fractions import Fraction as F
import numpy as np
import cv2
from omr_event_records import records
from align_homr_geometry import _position
from omr_bar_geometry import measure_systems,bar_bounds
from repair_grace_timing import _rebuild
from repair_tuplets import continuous_beams
from .relation_group_clock import bounded_map
from .relation_clock_lattice import written
from .relation_hollow_evidence import enclosed_hollow


def _groups(rs):
    lanes=defaultdict(list)
    for r in rs:
        lane=lanes[r['staff'],r['voice']]
        if r['chord']:
            if not lane:return None
            lane[-1].append(r)
        else:lane.append([r])
    return lanes


def _head_proof(h,n,image):
    p=_position(n)
    if p is None or h['position']!=p[1]:return False
    s=float(h['staff_space']);x,y=float(h['x']),float(h['y'])
    if not 5<=s<=80:return False
    l,r=round(x-.3*s),round(x+.3*s);t,b=round(y-.15*s),round(y+.15*s)
    if min(l,t)<0 or r>=image.shape[1] or b>=image.shape[0]:return False
    if n.findtext('type') in ('whole','half'):return enclosed_hollow(dict(h,xy=[x,y]),image,s)
    return (image[t:b+1,l:r+1]<140).mean()>=.85


def _duration_proof(r):
    n=r['node'];value=written(n)
    if value is None:return False
    tm=n.find('time-modification')
    if tm is not None:
        actual,normal=tm.findtext('actual-notes'),tm.findtext('normal-notes')
        if not actual or not normal or not actual.isdecimal() or not normal.isdecimal() or not 1<=int(actual)<=16 or not 1<=int(normal)<=16:return False
        value*=F(int(normal),int(actual))
    return r['duration']==value


def _physical_bars(root,homr,image):
    systems=measure_systems(root);result={};offset=0
    if len(systems)!=len(homr.get('systems',[])):return result
    for ms,sy in zip(systems,homr['systems']):
        if len(sy['staffs'])!=1:offset+=len(ms);continue
        staff=sy['staffs'][0];bounds=bar_bounds(staff,image)
        if bounds is not None and len(bounds)==len(ms)+1:
            for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
                heads=[h for h in staff['notes'] if lo<h['x']<hi]
                if 1<=len(heads)<=128:result[offset+j]=heads
        offset+=len(ms)
    return result


def _beam_proof(seq,mapped,ink,native_pair=None):
    """Every short written event must connect to an adjacent printed beam."""
    short={'eighth':1,'16th':2,'32nd':3};hits=set()
    for i,(a,b) in enumerate(zip(seq,seq[1:])):
        aa,bb=a[3],b[3];number=short.get(aa[0]['node'].findtext('type'))
        if number is None or number!=short.get(bb[0]['node'].findtext('type')):continue
        for up in (True,False):
            pick=min if up else max
            ha=pick([mapped[r['node']] for r in aa],key=lambda h:h['y']);hb=pick([mapped[r['node']] for r in bb],key=lambda h:h['y'])
            if continuous_beams(ha,hb,ink,up,number,maximum_gap=8) or native_pair is not None and native_pair(ha,hb,up,number):hits.update((i,i+1));break
    return all(i in hits for i,g in enumerate(seq) if g[3][0]['node'].findtext('type') in short)


class NativeBeamProof:
    """Lazy native beam ROIs after exact captured-raster reproduction.

    ``a`` and ``b`` use Homr preprocessed coordinates. The native gray image
    is loaded once; threshold masks are bounded pairwise regions. Failure to
    certify the photograph's crop/resize/CLAHE transform always abstains.
    """
    def __init__(self,geometry,image,photo,diagnostics=None):
        self.geometry=geometry;self.image=image;self.photo=photo
        self.diag=diagnostics if diagnostics is not None else Counter()
        self.native=None;self.checked=False;self.scale=None

    def _load(self):
        if self.checked:return self.native is not None
        self.checked=True
        if self.photo is None or self.image.ndim!=2 or self.image.size>16000000 or max(self.image.shape)>8192:return False
        if self.geometry.get('preprocessed_size')!=[self.image.shape[1],self.image.shape[0]]:return False
        try:
            from PIL import Image
            from homr.autocrop import autocrop
            from homr.resize import resize_image,calc_target_image_size
            from homr.color_adjust import apply_clahe
            with Image.open(self.photo) as meta:
                if meta.width*meta.height>16000000 or max(meta.size)>8192:return False
            color=cv2.imread(str(self.photo))
            if color is None:return False
            cropped=autocrop(color)
            target=calc_target_image_size(cropped.shape[1],cropped.shape[0])
            # Fixed-width resize can otherwise amplify a narrow long input
            # before the later exact-raster guard gets a chance to abstain.
            if target!=tuple(self.geometry['preprocessed_size']) or target[0]*target[1]>16000000 or max(target)>8192:return False
            reproduced=apply_clahe(resize_image(cropped))
            if reproduced.shape!=self.image.shape or not np.array_equal(reproduced,self.image):return False
            native=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
            sx,sy=native.shape[1]/self.image.shape[1],native.shape[0]/self.image.shape[0]
            if abs(sx-sy)>.002*sx:return False
            self.native=native;self.scale=(sx,sy)
            self.diag['native_transform_exactly_certified']+=1
            return True
        except (OSError,ValueError,ImportError):return False

    def test(self,a,b,up,number):
        if number not in (1,2) or not self._load():return False
        native=self.native;sx,sy=self.scale;s=a['staff_space']*min(sx,sy)
        pair=[dict(x=(h['x']+.5)*sx-.5,y=(h['y']+.5)*sy-.5,staff_space=s) for h in (a,b)]
        left=max(0,round(min(h['x'] for h in pair)-s));right=min(native.shape[1],round(max(h['x'] for h in pair)+s)+1)
        top=max(0,round(min(h['y'] for h in pair)-9*s));bottom=min(native.shape[0],round(max(h['y'] for h in pair)+9*s)+1)
        if not 0<(right-left)*(bottom-top)<=4000000:return False
        tile=(native[top:bottom,left:right]<140).astype('uint8')*255
        local=[dict(h,x=h['x']-left,y=h['y']-top) for h in pair]
        self.diag['native_beam_pair_checks']+=1
        return continuous_beams(*local,tile,up,number,maximum_gap=8)


def fuse(root,homr,image,diagnostics=None,*,photo=None):
    diag=diagnostics if diagnostics is not None else Counter()
    if image.ndim!=2 or image.size>16000000 or max(image.shape)>8192 or len(root.findall('part'))!=1:return []
    if len(root.findall('part/measure'))>2048 or len(homr.get('systems',[]))>128 or sum(len(st.get('notes',[])) for sy in homr.get('systems',[]) for st in sy.get('staffs',[]))>8192:return []
    mapped=bounded_map(root,homr,image);physical=_physical_bars(root,homr,image);ink=(image<140).astype('uint8')*255;jobs=[];meter=None;division=1
    native=NativeBeamProof(homr,image,photo,diag)
    for mi,(m,rs) in enumerate(zip(root.findall('part/measure'),records(root))):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            time=a.find('time')
            if time is not None:
                b,u=time.findtext('beats'),time.findtext('beat-type')
                meter=F(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u) else None
        if not 1<=len(rs)<=128 or not 1<=division<=9600 or m.get('implicit')=='yes':continue
        first=next((j for j,n in enumerate(m) if n.tag=='note'),None)
        if first is None or any(n.tag in ('forward','backup') for n in list(m)[:first]):continue
        if any(r['midi'] is None or r['node'].get('print-object')=='no' for r in rs):continue
        bar_heads=physical.get(mi)
        if bar_heads is None:continue
        bindings=[mapped[r['node']] for r in rs if r['node'] in mapped]
        # Include auxiliary and grace notes in the bar's detector-head
        # bijection. Detected but unexported heads prevent a clock reset.
        if len(bindings)!=len(rs) or len({id(h) for h in bindings})!=len(bindings) or {id(h) for h in bindings}!={id(h) for h in bar_heads}:continue
        lanes=_groups([r for r in rs if r['duration']>0 and not r['grace']])
        if lanes is None or not 2<=len(lanes)<=4:continue
        if {r['staff'] for r in rs}!={1,2}:continue
        proposals=[];valid=True;totals=[]
        extra=[]
        for key,groups in lanes.items():
            if len(groups)==1:extra.extend(groups[0]);continue
            if not 2<=len(groups)<=64 or any(len(g)>8 for g in groups):valid=False;break
            clock=F();seq=[]
            for g in groups:
                dur=g[0]['duration']
                if any(r['duration']!=dur or r['onset']!=g[0]['onset'] or not _duration_proof(r) or r['node'] not in mapped or not _head_proof(mapped[r['node']],r['node'],image) for r in g):valid=False;break
                group_heads=[mapped[r['node']] for r in g];space=float(np.median([h['staff_space'] for h in group_heads]));xs=[h['x'] for h in group_heads]
                if max(xs)-min(xs)>.45*space:valid=False;break
                x=float(np.median(xs));seq.append((x,clock,space,g));clock+=dur
            if not valid:break
            if any(b[0]-a[0]<.65*min(a[2],b[2]) for a,b in zip(seq,seq[1:])):valid=False;break
            if not _beam_proof(seq,mapped,ink,native.test):valid=False;break
            totals.append(clock);proposals.append(seq)
        if not valid or len(extra)>2 or len(proposals)!=2 or {g[0][3][0]['staff'] for g in proposals}!={1,2} or len(set(totals))!=1 or not 1<=totals[0]<=16 or meter is not None and totals[0]!=meter:continue
        # A partial auxiliary lane is never serialized into either complete
        # clock. Both proved lanes start at the earliest visible bar column.
        earliest=min((h['x'] for h in bar_heads),default=None)
        if earliest is None or any(abs(seq[0][0]-earliest)>.4*seq[0][2] for seq in proposals):continue
        diag['complete_duration_lanes']+=1
        hits=[]
        for i,a in enumerate(proposals):
            for b in proposals[i+1:]:
                if a[0][3][0]['staff']==b[0][3][0]['staff']:continue
                paired=[]
                for ax,at,asp,ag in a:
                    close=[p for p in b if abs(p[0]-ax)<=.4*min(asp,p[2])]
                    if not close:continue
                    if len(close)!=1 or close[0][1]!=at:valid=False;break
                    paired.append(at)
                if not valid:break
                if len(set(paired))>=2 and 0 in paired:hits.append(len(set(paired)))
            if not valid:break
        if not valid or not hits:diag['photographic_column_unproven']+=1;continue
        desired={r['node']:int(r['onset']*division) for r in rs}
        desired.update({r['node']:int(t*division) for seq in proposals for x,t,s,g in seq for r in g})
        if any((t*division).denominator!=1 for seq in proposals for x,t,s,g in seq):continue
        changed=sum(r['onset']!=F(desired[r['node']],division) for r in rs)
        if not changed:continue
        jobs.append((mi,division,desired,changed,min(hits),str(totals[0]),meter is None))
    if len(jobs)>128:return []
    changes=[];ms=root.findall('part/measure')
    for mi,division,desired,changed,columns,beats,unknown in jobs:
        m=ms[mi];saved=copy.deepcopy(list(m))
        def identity(rs):return [(r['staff'],r['voice'],r['midi'],r['duration'],r['chord'],r['grace'],r['onset'] if r['grace'] else None) for r in rs]
        before_rs=records(root)[mi];before=identity(before_rs);expected_duration={r['node']:r['duration'] for r in before_rs}
        if not _rebuild(m,desired):m[:]=saved;continue
        actual=records(root)[mi]
        if len(actual)!=len(desired) or any(r['node'] not in desired or r['onset']!=F(desired[r['node']],division) or r['duration']!=expected_duration[r['node']] for r in actual):m[:]=saved;continue
        if identity(actual)!=before:raise ValueError('Clock reconstruction changed note identity or grace onset')
        changes.append(dict(measure=mi+1,changed_events=changed,coincident_columns=columns,bar_beats=beats,unknown_meter=unknown,method='complete_existing_duration_clocks_and_photographic_columns'))
    return changes
