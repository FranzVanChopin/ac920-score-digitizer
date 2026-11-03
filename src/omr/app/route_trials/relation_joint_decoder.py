"""Transactional physical rhythm candidates; no reference-score inputs."""
import copy
from collections import Counter
from fractions import Fraction as F
import xml.etree.ElementTree as E
from omr_event_records import records
from .relation_multistream_clock_trial import fuse as multistream_clock
from .relation_event_clock_trial import _duration_proof


def protected(root, written_changes=False):
    bars=[]
    for row in records(root):
        notes=Counter();fixed=Counter()
        for r in row:
            n=copy.deepcopy(r['node'])
            tags=('duration','time-modification','type','beam') if written_changes else ('duration','time-modification')
            if r['grace'] or r['midi'] is None:tags=('duration',)
            for tag in tags:
                for e in n.findall(tag):n.remove(e)
            for e in n.iter():e.tail=None;e.text=e.text.strip() if e.text else None
            key=E.tostring(n,encoding='unicode');notes[key]+=1
            if r['grace'] or r['midi'] is None:fixed[key,r['onset'],r['duration']]+=1
        bars.append((notes,fixed))
    return bars


def _clean(node):
    for e in node.iter():e.tail=None;e.text=e.text.strip() if e.text else None
    return E.tostring(node,encoding='unicode')


def metadata(root,targets):
    """Protect structure/metadata, comparing time offsets in beats."""
    result=copy.deepcopy(root)
    for part in result.findall('part'):
        division=1
        for mi,m in enumerate(part.findall('measure'),1):
            for child in list(m):
                if child.tag=='attributes':
                    for d in child.findall('divisions'):
                        division=int(d.text);d.text='#time-unit'
                        if division<=0:raise ValueError('Nonpositive divisions')
                if child.tag=='note' or mi in targets and child.tag in ('backup','forward'):
                    m.remove(child);continue
                for tag in ('duration','offset'):
                    for t in child.iter(tag):t.text=str(F(t.text)/division)
    return _clean(result)


def transaction_valid(before,after,changes):
    targets={r.get('measure') for r in changes}
    old=records(before);new=records(after)
    if not targets or any(type(m) is not int or not 1<=m<=len(old) for m in targets):return False
    if len(old)!=len(new) or metadata(before,targets)!=metadata(after,targets):return False
    if protected(before,True)!=protected(after,True):return False
    for mi,(a,b) in enumerate(zip(old,new),1):
        if len(a)!=len(b):return False
        for x,y in zip(a,b):
            # The current decoder rebuilds cursors in the existing note order.
            # Checking each identity prevents equal-pitch nodes being swapped.
            nx=copy.deepcopy(x['node']);ny=copy.deepcopy(y['node'])
            for n in (nx,ny):
                for tag in ('duration','time-modification','type','beam'):
                    for t in n.findall(tag):n.remove(t)
            if _clean(nx)!=_clean(ny):return False
            old_type=x['node'].findtext('type')
            fields=('type','time-modification','beam','notations')
            written_changed=any([_clean(copy.deepcopy(e)) for e in x['node'].findall(key)]!=[_clean(copy.deepcopy(e)) for e in y['node'].findall(key)] for key in fields)
            if mi not in targets or old_type not in ('eighth','16th'):
                # Even wrong events outside the declared jobs must be fixed.
                if x['duration']!=y['duration'] or (mi not in targets or old_type!='quarter') and x['onset']!=y['onset'] or written_changed:return False
            elif x['duration']!=y['duration'] or written_changed:
                if y['node'].findtext('type') not in ('eighth','16th') or not _duration_proof(y):return False
    return True


def fuse(root,geometry,image,*,photo=None):
    report={'enabled':True,'stages':{},'reference_supplied':False,'no_additional_model_or_PL':True,
            'maximum_events_per_bar':128,'maximum_physical_slots':48,'maximum_ratio_plans':9,
            'maximum_native_pixels':16000000,'maximum_native_roi_pixels':4000000}
    # The independent native experiment remains a diagnostic. Its changes
    # overlap this stage, so running both would add cost without a proven gain.
    for name,fn,written_changes in [('joint_multistream',multistream_clock,True)]:
        row={'changes':[]};report['stages'][name]=row
        try:
            candidate=copy.deepcopy(root);diag=Counter()
            changes=fn(candidate,geometry,image,diag,photo=photo)
            if changes:
                if not transaction_valid(root,candidate,changes):
                    raise ValueError('Physical clock decoder changed protected note or fixed-event identities')
                root[:]=list(candidate);root.attrib.clear();root.attrib.update(candidate.attrib)
                root.text=candidate.text;root.tail=candidate.tail;row['changes']=changes
            row['diagnostics']=dict(diag)
        except Exception as exc:row.update(fallback=True,error=f'{type(exc).__name__}: {exc}')
    return report
