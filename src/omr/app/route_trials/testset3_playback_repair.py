"""Input-only, offset-aware XML serialization repair; no reference/model use.

The safe normalizer adjusts a following backup by the rest-duration delta,
not to a guessed zero. Its transaction proves every original note onset and
every non-normalized note duration unchanged. The separate final-XML recovery
is deliberately narrower: a negative return to an existing voice with one
complete measure lane and an adjacent meter-sized whole-measure rest.
"""
import copy
from fractions import Fraction
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

MAXIMUM_XML_BYTES = 32_000_000
MAXIMUM_MEASURES = 10_000
MAXIMUM_NOTES = 100_000
MAXIMUM_RECOVERY_NOTES = 128


def _local(node):
    return node.tag.rsplit('}', 1)[-1]


def _children(node, name):
    return [n for n in node if _local(n) == name]


def _child(node, name):
    return next((n for n in node if _local(n) == name), None)


def _text(node, name, default=''):
    n = _child(node, name)
    return n.text.strip() if n is not None and n.text else default


def _fingerprint(node):
    clone = copy.deepcopy(node)
    clone.tail = None
    for n in clone.iter():
        if n.text and not n.text.strip(): n.text = None
        if n.tail and not n.tail.strip(): n.tail = None
    return hashlib.sha256(ET.tostring(clone, encoding='utf8')).hexdigest()


def _limits(root):
    if _local(root) != 'score-partwise': raise ValueError('Expected score-partwise XML')
    if sum(_local(n) == 'measure' for n in root.iter()) > MAXIMUM_MEASURES:
        raise ValueError('Playback repair measure budget exceeded')
    if sum(_local(n) == 'note' for n in root.iter()) > MAXIMUM_NOTES:
        raise ValueError('Playback repair note budget exceeded')


def _timeline(root):
    """Permissive exact-fraction trace; negative cursors are evidence, not clamped."""
    result, moves = {}, []
    for pi, part in enumerate(_children(root, 'part')):
        divisions = 1
        for mi, measure in enumerate(_children(part, 'measure')):
            cursor = previous = Fraction(0); ni = 0
            for ci, node in enumerate(measure):
                tag = _local(node)
                if tag == 'attributes':
                    divisions = int(_text(node, 'divisions', str(divisions)))
                    if divisions <= 0: raise ValueError('Positive divisions required')
                elif tag in ('backup', 'forward'):
                    duration = Fraction(_text(node, 'duration', '0')) / divisions
                    before = cursor
                    cursor += duration if tag == 'forward' else -duration
                    if cursor < 0:
                        moves.append(dict(part_index=pi, measure_index=mi, child_index=ci,
                                          before=str(before), after=str(cursor)))
                elif tag == 'note':
                    chord = _child(node, 'chord') is not None
                    grace = _child(node, 'grace') is not None
                    onset = previous if chord else cursor
                    duration = Fraction(_text(node, 'duration', '0')) / divisions
                    if not chord: previous = onset
                    if not chord and not grace: cursor += duration
                    result[pi, mi, ni] = dict(onset=onset, duration=duration, staff=_text(node, 'staff', '1'),
                        voice=_text(node, 'voice', '1'), grace=grace, chord=chord, hash=_fingerprint(node),
                        rest=_child(node, 'rest') is not None)
                    ni += 1
    return result, moves


def _commit(root, working):
    root.clear(); root.attrib.update(working.attrib)
    root.text, root.tail = working.text, working.tail
    root.extend(list(working))


def normalize_measure_rests_safe(root, original_normalizer, initial_meter=None):
    """Transactional wrapper for board_homr.normalize_measure_rests.

    Return JSON diagnostics with normalized_rests and changes. Keep the
    original normalizer itself unchanged; parent integration owns that call.
    """
    _limits(root)
    before, _ = _timeline(root)
    working = copy.deepcopy(root)
    captured = []
    for pi, part in enumerate(_children(working, 'part')):
        for mi, measure in enumerate(_children(part, 'measure')):
            nodes = list(measure); ni = 0
            for ci, node in enumerate(nodes):
                if _local(node) != 'note': continue
                rest = _child(node, 'rest')
                if rest is not None and (rest.get('measure') == 'yes' or _text(node, 'type') == 'whole'):
                    duration = _child(node, 'duration')
                    following = nodes[ci+1] if ci+1 < len(nodes) else None
                    backup = _child(following, 'duration') if following is not None and _local(following) == 'backup' else None
                    if duration is not None and backup is not None:
                        captured.append((pi, mi, ni, duration, int(duration.text), backup, int(backup.text)))
                ni += 1
    normalized = original_normalizer(working, initial_meter)
    changes = []
    for pi, mi, ni, duration, old_duration, backup, old_backup in captured:
        new_duration = int(duration.text)
        if new_duration == old_duration: continue
        desired = old_backup + new_duration - old_duration
        if desired < 0:
            return dict(applied=False, normalized_rests=0, proposed_normalized_rests=normalized, changes=[],
                        abstain_reason='duration_delta_requires_forward_not_backup', input_unchanged=True)
        if int(backup.text) != desired:
            backup.text = str(desired)
            changes.append(dict(part_index=pi, measure_index=mi, note_index=ni,
                old_rest_units=old_duration, new_rest_units=new_duration, old_backup_units=old_backup,
                new_backup_units=desired, return_offset_preserved=True))
    after, _ = _timeline(working)
    assert set(before) == set(after), 'Normalizer changed note cardinality'
    normalized_rest_keys = {key for key in before if before[key]['rest'] and before[key]['duration'] != after[key]['duration']}
    for key in before:
        a, b = before[key], after[key]
        if a['onset'] != b['onset'] or any(a[k] != b[k] for k in ('staff','voice','grace','chord')):
            return dict(applied=False, normalized_rests=0, proposed_normalized_rests=normalized, changes=[],
                        abstain_reason='original_note_onset_or_identity_would_change', input_unchanged=True)
        if key not in normalized_rest_keys and (a['duration'] != b['duration'] or a['hash'] != b['hash']):
            return dict(applied=False, normalized_rests=0, proposed_normalized_rests=normalized, changes=[],
                        abstain_reason='non_normalized_note_content_would_change', input_unchanged=True)
    _commit(root, working)
    return dict(applied=True, normalized_rests=normalized, changes=changes,
        note_count=len(after), normalized_rest_count=len(normalized_rest_keys),
        original_staff_voice_onsets_preserved=True, original_pitched_note_subtrees_and_durations_preserved=True,
        original_non_normalized_rest_subtrees_and_durations_preserved=True,
        model_calls=0, reference_used=False)


def repair_negative_backups(root, geometry=None):
    """Recover only a witnessed complete returning lane in already-final XML.

    This changes corrupted suffix onsets explicitly. It does not claim to
    preserve their invalid old timestamps or independently prove correctness.
    The safe upstream normalizer above is the preferred production repair.
    """
    _limits(root)
    working = copy.deepcopy(root)
    before, negative = _timeline(working)
    changes, abstentions = [], []
    for pi, part in enumerate(_children(working, 'part')):
        divisions = 1; meter = None
        for mi, measure in enumerate(_children(part, 'measure')):
            nodes = list(measure); cursor = Fraction(0); tails = {}; note_keys = {}; ni = 0
            for ci, node in enumerate(nodes):
                tag = _local(node)
                if tag == 'attributes':
                    divisions = int(_text(node, 'divisions', str(divisions)))
                    time = _child(node, 'time')
                    if time is not None:
                        meter = Fraction(sum(int(x) for x in _text(time,'beats','4').split('+'))*4,int(_text(time,'beat-type','4')))
                elif tag == 'note':
                    key = (_text(node,'voice','1'), _text(node,'staff','1'))
                    if _child(node,'chord') is None and _child(node,'grace') is None:
                        cursor += Fraction(_text(node,'duration','0'))/divisions
                        tails[key] = cursor
                    note_keys[ci] = (pi,mi,ni); ni += 1
                elif tag in ('forward','backup'):
                    duration = Fraction(_text(node,'duration','0'))/divisions
                    if tag == 'forward': cursor += duration; continue
                    if cursor-duration >= 0: cursor -= duration; continue
                    previous = nodes[ci-1] if ci else None
                    suffix = nodes[ci+1:]
                    reason = None
                    if (meter is None or len(suffix) == 0 or len(suffix) > MAXIMUM_RECOVERY_NOTES
                        or previous is None or _local(previous) != 'note'
                        or _child(previous,'rest') is None or _child(previous,'rest').get('measure') != 'yes'
                        or _child(previous,'chord') is not None or _child(previous,'grace') is not None
                        or before.get(note_keys.get(ci-1),{}).get('onset') != 0
                        or cursor != meter or Fraction(_text(previous,'duration','0'))/divisions != meter):
                        reason = 'no_adjacent_meter_sized_whole_rest_witness'
                    elif any(_local(n) != 'note' or _child(n,'chord') is not None or _child(n,'grace') is not None for n in suffix):
                        reason = 'suffix_not_one_plain_note_lane'
                    else:
                        keys = {(_text(n,'voice','1'),_text(n,'staff','1')) for n in suffix}
                        lane = next(iter(keys)) if len(keys) == 1 else None
                        tail = tails.get(lane)
                        if lane is None or tail is None or tail < 0 or tail > meter or lane[1] == _text(previous,'staff','1'):
                            reason = 'returning_lane_not_uniquely_witnessed_on_partner_staff'
                        elif tail + sum((Fraction(_text(n,'duration','0'))/divisions for n in suffix),Fraction(0)) != meter:
                            reason = 'returning_lane_not_complete_to_same_printed_meter'
                    if reason:
                        abstentions.append(dict(part_index=pi,measure_index=mi,child_index=ci,reason=reason))
                        cursor -= duration; continue
                    replacement = (cursor-tail)*divisions
                    if replacement.denominator != 1 or replacement < 0:
                        abstentions.append(dict(part_index=pi,measure_index=mi,child_index=ci,reason='noninteger_return_offset'))
                        cursor -= duration; continue
                    _child(node,'duration').text = str(replacement.numerator)
                    changes.append(dict(part_index=pi,measure_index=mi,child_index=ci,
                        old_backup_units=int(duration*divisions),new_backup_units=replacement.numerator,
                        returning_voice=lane[0],returning_staff=lane[1],previous_lane_tail_q=str(tail),
                        meter_q=str(meter),suffix_notes=len(suffix),
                        reason='complete_existing_lane_return_after_normalized_partner_whole_rest'))
                    cursor = tail
    after, remaining = _timeline(working)
    assert set(before) == set(after)
    changed_onsets = []
    affected = {(r['part_index'],r['measure_index']) for r in changes}
    for key in before:
        a,b = before[key],after[key]
        assert a['hash'] == b['hash'] and a['duration'] == b['duration']
        if a['onset'] != b['onset']:
            assert key[:2] in affected
            changed_onsets.append(dict(part_index=key[0],measure_index=key[1],note_index=key[2],
                                      before_q=str(a['onset']),after_q=str(b['onset']),staff=b['staff'],voice=b['voice']))
    if changes: _commit(root, working)
    return dict(changes=changes,abstentions=abstentions,changed_onsets=changed_onsets,
        invalid_cursor_moves_before=len(negative),invalid_cursor_moves_after=len(remaining),
        original_note_subtrees_staff_voice_pitch_duration_preserved=True,
        unaffected_measure_onsets_preserved=True,changed_suffix_onsets_not_claimed_originally_correct=True,
        geometry_used=False,model_calls=0,reference_used=False)


def repair_file(input_xml, output_xml, geometry=None):
    """Write a distinct experimental artifact; never modify a baseline file."""
    source, target = Path(input_xml), Path(output_xml)
    if source.resolve() == target.resolve() or target.exists(): raise FileExistsError('Use a new distinct XML output')
    if source.stat().st_size > MAXIMUM_XML_BYTES: raise ValueError('XML budget exceeded')
    raw = source.read_bytes(); root = ET.fromstring(raw)
    diagnostics = repair_negative_backups(root,geometry)
    target.parent.mkdir(parents=True,exist_ok=True)
    data = ET.tostring(root,encoding='utf8',xml_declaration=True)
    target.write_bytes(data)
    diagnostics.update(input_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=hashlib.sha256(data).hexdigest(),
                       baseline_modified=False,output=str(target))
    return diagnostics
