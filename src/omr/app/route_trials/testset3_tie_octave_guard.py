"""Keep new optical ties only when octave-refined sounding pitches agree.

Capture existing mark identities BEFORE recovery, bind accepted real note
objects immediately AFTER recovery, then validate AFTER octave/attribute
changes. No current child index is used during validation. Old mark objects
are never removed; missing provenance produces diagnostics only.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import xml.etree.ElementTree as ET

MAXIMUM_BOUND_PAIRS=512
PITCH={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}


@dataclass(frozen=True)
class _Existing:
    owner: ET.Element
    marks: frozenset


@dataclass(frozen=True)
class _Pair:
    first: ET.Element
    last: ET.Element
    info: dict
    new_marks: tuple


@dataclass(frozen=True)
class _Guard:
    owner: ET.Element
    pairs: tuple
    binding_errors: tuple
    provenance_known: bool


def _marks(note,kind=None):
    for parent in (note,note.find('notations')):
        if parent is None:continue
        for mark in parent:
            if mark.tag in ('tie','tied') and (kind is None or mark.get('type')==kind):
                yield parent,mark


def capture_existing(root):
    """Opaque identity snapshot; take before the frozen recovery adds tags."""
    return _Existing(root,frozenset(mark for note in root.findall('part/measure/note') for _,mark in _marks(note)))


def _midi(note):
    pitch=note.find('pitch')
    try:
        return Fraction(12*(int(pitch.findtext('octave'))+1)+PITCH[pitch.findtext('step')])+Fraction(pitch.findtext('alter','0'))
    except (AttributeError,KeyError,TypeError,ValueError,ZeroDivisionError):return None


def bind(root,tie_report,*,existing=None):
    """Bind live ET endpoints before octave recovery can insert attributes.

    Supply ``existing=capture_existing(root)`` taken BEFORE tie recovery. When
    omitted, this guard deliberately cannot delete any tag of unknown origin.
    """
    if existing is not None and (not isinstance(existing,_Existing) or existing.owner is not root):
        raise ValueError('Existing tie snapshot belongs to a different score tree')
    candidates=tie_report.get('accepted_pairs',[])
    if len(candidates)>MAXIMUM_BOUND_PAIRS:raise ValueError('Tie guard exceeds 512 pairs')
    pairs=[];errors=[];seen=set();parts=root.findall('part')
    for info in candidates:
        try:
            part=parts[info['part']];measures=part.findall('measure')
            first=list(measures[info['start_measure']-1])[info['first_xml_child_index']]
            last=list(measures[info['stop_measure']-1])[info['last_xml_child_index']]
            if first.tag!='note' or last.tag!='note':raise ValueError('Bound child is not a note')
            expected=Fraction(info['midi'])
            if any(_midi(n)!=expected for n in (first,last)):raise ValueError('Bound note pitch differs from recovered endpoint')
            for n in (first,last):
                if n.findtext('staff','1')!=str(info['staff']) or n.findtext('voice','1')!=str(info['voice']):
                    raise ValueError('Bound staff/voice differs from recovered endpoint')
            key=(id(first),id(last))
            if key in seen:continue
            seen.add(key)
            new=[]
            if existing is not None:
                for note,kind in ((first,'start'),(last,'stop')):
                    new.extend((parent,mark) for parent,mark in _marks(note,kind) if mark not in existing.marks)
            public={k:info[k] for k in ('part','staff','voice','midi','start_measure','stop_measure','start_onset','stop_onset') if k in info}
            pairs.append(_Pair(first,last,public,tuple(new)))
        except (IndexError,KeyError,TypeError,ValueError,ZeroDivisionError) as exc:
            errors.append({'endpoint':{k:info.get(k) for k in ('part','start_measure','stop_measure','midi')},'error':str(exc)})
    return _Guard(root,tuple(pairs),tuple(errors),existing is not None)


def validate_after_octave(root,guard):
    """Remove only tracked new marks when bound sounding pitches disagree."""
    if not isinstance(guard,_Guard) or guard.owner is not root:
        raise ValueError('Tie guard belongs to a different score tree')
    live=set(root.findall('part/measure/note'));rows=[];removed=0;invalid=0;unbound=0;valid=0
    for pair in guard.pairs:
        info=dict(pair.info)
        if pair.first not in live or pair.last not in live:
            unbound+=1;rows.append(dict(info,status='unverified',reason='bound_note_replaced_or_detached',removed_marks=0));continue
        a,b=_midi(pair.first),_midi(pair.last)
        if a is not None and a==b:
            valid+=1;rows.append(dict(info,status='valid',sounding_midi=[str(a),str(b)],removed_marks=0));continue
        invalid+=1;count=0
        if guard.provenance_known:
            for parent,mark in pair.new_marks:
                if mark in list(parent):parent.remove(mark);count+=1
        removed+=count
        rows.append(dict(info,status='revoked_new_marks' if guard.provenance_known else 'unverified_provenance',
            reason='sounding_pitch_missing' if a is None or b is None else 'sounding_pitch_mismatch_after_octave',
            sounding_midi=[str(a) if a is not None else None,str(b) if b is not None else None],removed_marks=count))
    return dict(algorithm='bound_optical_tie_sounding_pitch_guard_v1',bound_pairs=len(guard.pairs),
                valid_sounding_pitch_pairs=valid,invalid_sounding_pitch_pairs=invalid,
                replaced_or_detached_note_pairs=unbound,removed_new_tie_or_tied_elements=removed,
                existing_mark_provenance_known=guard.provenance_known,original_marks_removed=0,
                child_indices_used_after_octave=False,reference_used=False,model_used=False,
                binding_errors=list(guard.binding_errors),pairs=rows)
