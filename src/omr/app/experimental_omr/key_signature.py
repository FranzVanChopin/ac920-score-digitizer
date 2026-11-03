"""Conservative printed key-signature recognition for treble-led systems.

The general accidental detector has many responses inside clefs and chords.
Only the short strip after the clef, before the first note, is eligible here.
Vertical positions and left-to-right order must agree with music notation.
An uncertain result remains ``None`` so it can be reviewed or overridden.
"""

from __future__ import annotations

from .detect import (GlyphSet, SYM_REST_HALF, SYM_REST_QUARTER,
                     SYM_REST_WHOLE)


# Accidental centroid positions measured in staff spaces below the top line.
# A flat's body is centred half a space above the pitch it alters.
_FLAT_Y = (1.5, 0.0, 2.0, 0.5, 3.0, 1.0, 3.5)
_SHARP_Y = (0.0, 1.5, -1.0, 0.5, 2.0, -0.5, 1.0)


def _sequence(candidates, staff, symbol: str, ys: tuple[float, ...],
              min_score: float, y_tolerance: float = 0.34):
    matches = []
    previous_x = None
    for offset in ys:
        choices = [a for a in candidates
                   if a.symbol == symbol and a.score >= min_score
                   and abs((8.0 - staff.y_to_step(a.y, a.x)) / 2.0
                           - offset) <= y_tolerance
                   and (previous_x is None
                        or 0.68 * staff.space <= a.x - previous_x
                        <= 1.48 * staff.space)]
        if not choices:
            break
        chosen = max(choices, key=lambda a: a.score)
        matches.append(chosen)
        previous_x = chosen.x
    return matches


def infer_treble_key(glyphs: GlyphSet) -> tuple[int | None, float]:
    """Return signed fifths and evidence score, or ``None`` if ambiguous."""
    staff = glyphs.staff
    bbox = glyphs.clef_bbox
    if staff is None or bbox is None or glyphs.clef != "treble":
        return None, 0.0
    space = staff.space
    right = bbox[0] + bbox[2]
    first = min((head.x for head in glyphs.noteheads()
                 if head.x > right + 0.65 * space), default=float("inf"))
    limit = min(right + 4.6 * space, first - 0.45 * space)
    if limit <= right + 0.6 * space:
        return None, 0.0
    candidates = [a for a in glyphs.raw_accidentals
                  if right + 0.6 * space < a.x < limit]

    flats = _sequence(candidates, staff, "accidental_flat", _FLAT_Y, 0.69)
    sharps = _sequence(candidates, staff, "accidental_sharp", _SHARP_Y,
                       0.56, y_tolerance=0.45)
    # A single weak sharp may be a staff remnant or a note ornament; two
    # ordered sharp symbols have independent pitch and spacing evidence.
    if len(sharps) == 1:
        rival = max((a.score for a in candidates
                     if a.symbol == "accidental_flat"
                     and abs(a.x - sharps[0].x) <= 0.4 * space
                     and abs(a.y - sharps[0].y) <= 0.55 * space), default=0.0)
        if sharps[0].score < 0.72 or sharps[0].score - rival < 0.04:
            sharps = []
    if len(flats) == 1 and flats[0].score < 0.70:
        flats = []
    if not flats and not sharps:
        return None, 0.0
    if len(flats) > len(sharps):
        return -len(flats), min(a.score for a in flats)
    if len(sharps) > len(flats):
        return len(sharps), min(a.score for a in sharps)
    flat_score = min(a.score for a in flats)
    sharp_score = min(a.score for a in sharps)
    if abs(flat_score - sharp_score) < 0.06:
        return None, 0.0
    return (-len(flats), flat_score) if flat_score > sharp_score else (len(sharps), sharp_score)


def infer_inline_key(glyphs: GlyphSet, barline: float) -> tuple[int | None, float]:
    """Find a newly printed key immediately after an interior barline.

    A genuine inline key reserves several staff spaces before the first note.
    A single accidental next to that note should not change the key.
    """
    staff = glyphs.staff
    if staff is None or glyphs.clef != "treble":
        return None, 0.0
    space = staff.space
    content_symbols = {"notehead_filled", "notehead_hollow",
                       SYM_REST_WHOLE, SYM_REST_HALF, SYM_REST_QUARTER}
    first = min((d.x for d in glyphs.detections
                 if d.symbol in content_symbols and d.x > barline + 0.4 * space),
                default=float("inf"))
    if first - barline < 3.2 * space:
        return None, 0.0
    candidates = [a for a in glyphs.raw_accidentals
                  if barline + 0.35 * space < a.x
                  < min(barline + 2.6 * space, first - 0.55 * space)]
    flats = _sequence(candidates, staff, "accidental_flat", _FLAT_Y, 0.64)
    sharps = _sequence(candidates, staff, "accidental_sharp", _SHARP_Y, 0.55)
    if flats and flats[0].x <= barline + 1.5 * space and flats[0].score >= 0.65:
        # The first symbol of a printed key starts near the barline.  Later
        # accidental responses at note positions cannot seed a key change.
        return -len(flats), min(a.score for a in flats)
    if (len(sharps) >= 2 and sharps[0].x <= barline + 1.5 * space
            and sharps[0].score >= 0.65):
        return len(sharps), min(a.score for a in sharps)
    if (len(sharps) == 1 and sharps[0].x <= barline + 1.5 * space
            and sharps[0].score >= 0.80):
        return 1, sharps[0].score
    return None, 0.0


def inline_key_changes(glyph_sets: list[GlyphSet],
                       systems: list[list[int]]) -> dict[int, tuple[tuple[float, int], ...]]:
    """Share treble-led inline key changes with every staff of each system."""
    result: dict[int, tuple[tuple[float, int], ...]] = {}
    for group in systems:
        treble = next((glyph_sets[i] for i in group
                       if glyph_sets[i].clef == "treble"), None)
        changes = []
        if treble is not None:
            for barline in treble.barlines[:-1]:
                key, _score = infer_inline_key(treble, barline)
                if key is not None:
                    changes.append((barline, key))
        for index in group:
            result[index] = tuple(changes)
    return result


def has_terminal_flat_cancellation(glyph_sets: list[GlyphSet],
                                   group: list[int]) -> tuple[bool, float]:
    """Find matching B-natural cancellation signs after the last piano bar.

    The first natural in a move from one flat to C major can appear as a
    courtesy key signature at the far right of the preceding system.  The
    next system then has no printed key symbols, so ordinary carry-forward
    would incorrectly retain B-flat for every following measure.
    """
    if len(group) != 2:
        return False, 0.0
    candidates = []
    for index in group:
        glyphs = glyph_sets[index]
        staff = glyphs.staff
        if staff is None or not glyphs.barlines:
            return False, 0.0
        bar = glyphs.barlines[-1]
        space = staff.space
        natural = [a for a in glyphs.raw_accidentals
                   if a.symbol == "accidental_natural" and a.score >= .59
                   and bar + .55 * space <= a.x <= bar + 1.85 * space
                   and a.x <= staff.x1 + .25 * space
                   and abs((a.y - staff.y_top_line) / space - 1.75) <= .4]
        if not natural:
            return False, 0.0
        candidates.append(max(natural, key=lambda a: a.score))
    space = sum(glyph_sets[i].staff.space for i in group) / 2
    return (abs(candidates[0].x - candidates[1].x) <= .5 * space,
            min(a.score for a in candidates))


def resolve_system_keys(glyph_sets: list[GlyphSet],
                        systems: list[list[int]],
                        override: int | None = None) -> list[int]:
    """Use treble evidence for each system; fill only bracketed missing keys."""
    if override is not None:
        if not -7 <= override <= 7:
            raise ValueError("key_fifths must be between -7 and 7")
        for glyphs in glyph_sets:
            glyphs.key_fifths = override
            glyphs.key_score = 1.0
        return [override] * len(systems)

    observed: list[int | None] = []
    scores: list[float] = []
    for group in systems:
        treble = next((glyph_sets[i] for i in group
                       if glyph_sets[i].clef == "treble"), None)
        key, score = infer_treble_key(treble) if treble else (None, 0.0)
        observed.append(key)
        scores.append(score)
    active = None
    for index, group in enumerate(systems[:-1]):
        if observed[index] is not None:
            active = observed[index]
        if active == -1 and observed[index + 1] is None:
            cancelled, evidence = has_terminal_flat_cancellation(glyph_sets, group)
            if cancelled:
                observed[index + 1] = 0
                scores[index + 1] = evidence
    resolved = observed[:]
    for index, key in enumerate(observed):
        if key is not None:
            continue
        left = next((k for k in reversed(observed[:index]) if k is not None), None)
        right = next((k for k in observed[index + 1:] if k is not None), None)
        if left is not None and left == right:
            resolved[index] = left
    # A printed key remains active until a new one appears.  Keep that state
    # when an intermediate system is too faint to classify; its score stays
    # zero in the review report to distinguish inference from direct reading.
    previous = None
    for index, key in enumerate(resolved):
        if key is not None:
            previous = key
        elif previous is not None:
            resolved[index] = previous
    values = [key if key is not None else 0 for key in resolved]
    for group, key, score in zip(systems, values, scores):
        for i in group:
            glyph_sets[i].key_fifths = key
            glyph_sets[i].key_score = score
    return values
