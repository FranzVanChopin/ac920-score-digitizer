"""Correct Homr 0.7's alternating tremolo serializer without changing time.

Homr emits a single ``tremolo`` token, then its XML generator toggles a global
start/stop flag and hardcodes three marks. That flag is not a recognized
two-note relationship. Use this adapter only for Homr-derived XML. Explicit
double tremolos with their MusicXML duration evidence are preserved.
"""
from fractions import Fraction


TYPES = {'whole': 4, 'half': 2, 'quarter': 1, 'eighth': Fraction(1, 2),
         '16th': Fraction(1, 4), '32nd': Fraction(1, 8), '64th': Fraction(1, 16)}


def _double_duration(note, divisions):
    written = TYPES.get(note.findtext('type'))
    tm = note.find('time-modification')
    if written is None or tm is None or note.find('grace') is not None:
        return False
    dots = len(note.findall('dot'))
    written = Fraction(written) * (2 - Fraction(1, 2 ** dots))
    actual = Fraction(int(note.findtext('duration', '0')), divisions)
    # Only a simple, explicitly timed double tremolo is proven here. This
    # Homr-token adapter is not a general importer for nested double tremolos.
    return (tm.findtext('actual-notes') == '2' and
            tm.findtext('normal-notes') == '1' and actual == written / 2)


def repair(root):
    changes = []
    for part in root.findall('part'):
        divisions = 1
        for measure in part.findall('measure'):
            proven = set(); pending = {}
            for note in measure:
                if note.tag == 'attributes':
                    divisions = int(note.findtext('divisions', str(divisions)))
                if note.tag != 'note':
                    continue
                key = (note.findtext('staff', '1'), note.findtext('voice', '1'))
                trem = note.find('notations/ornaments/tremolo')
                if trem is None:
                    if note.find('chord') is None: pending.pop(key, None)
                    continue
                if trem.get('type') == 'start' and _double_duration(note, divisions):
                    pending[key] = (note, trem)
                elif trem.get('type') == 'stop' and key in pending and _double_duration(note, divisions):
                    start, first = pending.pop(key)
                    if first.text == trem.text and start.findtext('duration') == note.findtext('duration'):
                        proven.update((first, trem))
                else:
                    pending.pop(key, None)
            for ni, note in enumerate(measure.findall('note')):
                for trem in note.findall('notations/ornaments/tremolo'):
                    if trem.get('type') not in ('start', 'stop') or trem in proven:
                        continue
                    old = trem.get('type'); trem.set('type', 'single')
                    changes.append({'part': part.get('id'), 'measure': measure.get('number'),
                                    'note_index': ni, 'old_type': old, 'new_type': 'single',
                                    'marks': trem.text, 'method': 'homr_token_has_no_double_note_relation'})
    return changes
