"""Transactional late clock repairs; reference notation is not an input."""
import copy
import xml.etree.ElementTree as E
from collections import Counter
from omr_event_records import records
from .relation_polyrhythm_trial import fuse as slot_clock
from .relation_event_clock_trial import fuse as event_clock


def identity(root, allow_duration):
    bars=[]
    for rs in records(root):
        notes=Counter();fixed=Counter()
        for r in rs:
            n=copy.deepcopy(r['node'])
            for tag in ('duration', 'time-modification') if allow_duration else ('duration',):
                for e in n.findall(tag):n.remove(e)
            for e in n.iter():e.tail=None;e.text=e.text.strip() if e.text else None
            key=E.tostring(n,encoding='unicode')
            notes[key]+=1
            if not allow_duration or r['grace'] or r['midi'] is None:
                fixed[(key,r['duration'],r['onset'] if r['grace'] or r['midi'] is None else None)]+=1
        bars.append((notes,fixed))
    return bars


def fuse(root, geometry, image, *, photo=None):
    report={'enabled':True,'stages':{},'reference_supplied':False,
            'maximum_events_per_bar':128,'maximum_time_slots':48,
            'maximum_native_pixels':16000000,'maximum_native_beam_roi_pixels':4000000,
            'global_meter_invented':False,'no_additional_model_or_PL':True}
    for name,fn,allow_duration in [('physical_empty_slots',slot_clock,True),
                                   ('complete_event_clocks',event_clock,False)]:
        row={'changes':[]};report['stages'][name]=row
        try:
            candidate=copy.deepcopy(root);diagnostics=Counter()
            changes=fn(candidate,geometry,image,diagnostics,**({'photo':photo} if name=='complete_event_clocks' else {}))
            if changes:
                if identity(root,allow_duration)!=identity(candidate,allow_duration):
                    raise ValueError('clock repair changed protected note identities or fixed events')
                root[:]=list(candidate);root.attrib.clear();root.attrib.update(candidate.attrib)
                root.text=candidate.text;root.tail=candidate.tail
                row['changes']=changes
            row['diagnostics']=dict(diagnostics)
        except Exception as exc:
            # A failed optional stage never mutates the last accepted score.
            row.update(fallback=True,error=f'{type(exc).__name__}: {exc}')
    return report
