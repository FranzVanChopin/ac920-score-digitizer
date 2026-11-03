"""Correct low bass octave partners only with complete image chord columns."""
from collections import Counter
import copy
import xml.etree.ElementTree as ET

from align_homr_geometry import _xml_systems,_position
from omr_bar_geometry import anchored_bar_bounds,measure_systems
from repair_ledger_chords import _groups,_shared_stem
from repair_pitch_context import _ink,no_accidental
from repair_unison_voices import _interior


def repair(root,geometry,image):
    systems=measure_systems(root);_xml_systems(root)
    if len(systems)!=len(geometry.get('systems',[])):return []
    clefs=root.findall('part/measure/attributes/clef')
    if not clefs or any(c.findtext('sign')!='F' or c.findtext('line')!='4'
                        or c.findtext('clef-octave-change','0')!='0' for c in clefs):return []
    ink=_ink(image);key=0;changes=[]
    for ms,system in zip(systems,geometry['systems']):
        if len(system.get('staffs',[]))!=1:continue
        staff=system['staffs'][0]
        bounds=anchored_bar_bounds(staff,image)
        if bounds is not None and len(bounds)!=len(ms)+1:bounds=None
        for i,m in enumerate(ms):
            for attr in m.findall('attributes'):key=int(attr.findtext('key/fifths',str(key)))
            if bounds is None:continue
            notes=m.findall('note')
            if (not notes or len({n.findtext('voice','1') for n in notes})!=1
                or any(n.find('pitch') is None or n.find('grace') is not None
                       or n.find('time-modification') is not None for n in notes)):continue
            raw=sorted((h for h in staff['notes'] if bounds[i]<h['x']<bounds[i+1]),key=lambda h:h['x'])
            groups=[]
            for h in raw:
                if not groups or h['x']-groups[-1][-1]['x']>3:groups.append([])
                groups[-1].append(h)
            xml_groups=_groups(notes)
            if len(groups)!=len(xml_groups) or any(len(a)!=len(b) for a,b in zip(groups,xml_groups)):continue
            # The full bar has exactly matching column counts. Each repaired
            # column additionally needs its own unchanged octave partner and
            # shared printed stem; uncertain columns remain untouched.
            plans=[]
            for ns,hs in zip(xml_groups,groups):
                pairs=list(zip(sorted(ns,key=lambda n:_position(n)[1]),sorted(hs,key=lambda h:h['position'])))
                wrong=[(n,h) for n,h in pairs if _position(n)[1]!=h['position']]
                if len(wrong)>1:continue
                if not wrong:continue
                n,h=wrong[0]
                if h['position']>-6 or abs(_position(n)[1]-h['position'])>6:continue
                partners=[(p,k) for p,k in pairs if p is not n and k['position']==h['position']+7
                          and _position(p)[1]==k['position']]
                if len(partners)!=1:continue
                p,k=partners[0]
                step=p.findtext('pitch/step');expected=(1 if step in 'FCGDAEB'[:max(key,0)] else
                                                     -1 if step in 'BEADGCF'[:max(-key,0)] else 0)
                if (n.find('accidental') is not None or p.find('accidental') is not None
                    or int(p.findtext('pitch/alter','0'))!=expected
                    or n.findtext('duration')!=p.findtext('duration')
                    or not _shared_stem(h,k) or (_interior(h,ink) or 0)<.7
                    or (_interior(k,ink) or 0)<.7 or not no_accidental(h,staff['notes'],ink)):
                    continue
                plans.append((n,p,h))
            for n,p,h in plans:
                pitch=n.find('pitch');old=ET.tostring(pitch,encoding='unicode')
                target=copy.deepcopy(p.find('pitch'));target.find('octave').text=str(int(target.findtext('octave'))-1)
                at=list(n).index(pitch);n.remove(pitch);n.insert(at,target)
                changes.append({'measure':m.get('number'),'method':'complete_low_octave_column',
                                'old_pitch':old,'step':target.findtext('step'),'octave':target.findtext('octave'),
                                'x':h['x'],'y':h['y']})
    return changes
