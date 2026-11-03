"""Retain segmentation heads beyond Homr's default four ledger-line spaces.

Only previously unassigned heads with stems, outside a unique nearest
staff, are reconsidered. Both upper and lower ledger notes compete for the
same head. No other model is loaded; original size/clump checks remain.
"""

from __future__ import annotations


def nearest_ledger_staff(staffs, x, y):
    """The closest physical staff must be uniquely farther than four spaces.

    Include nearby normal-range staves in the competition: an upper-only
    search can otherwise steal a previous row's low note for the next row.
    """
    candidates = []
    for staff in staffs:
        point = staff.get_at(x)
        if point is None or len(point.y) != 5 or point.average_unit_size <= 0:
            continue
        distance = max(point.y[0]-y, y-point.y[-1], 0)/point.average_unit_size
        candidates.append((distance, staff, point))
    candidates.sort(key=lambda item: item[0])
    if not candidates or not 4 < candidates[0][0] <= 8:
        return None
    if len(candidates) > 1 and candidates[1][0]-candidates[0][0] < 1.5:
        return None
    return candidates[0][1]


def install_extended_ledger_detection() -> None:
    import homr.main as main

    original = main.add_notes_to_staffs
    if getattr(original, "_omr_extended_ledger", False):
        return

    def add_with_ledger_heads(staffs, noteheads, symbols, notehead_pred):
        notes = original(staffs, noteheads, symbols, notehead_pred)
        for staff in staffs:
            for note in staff.get_notes():
                note._omr_staff_space = float(staff.average_unit_size)
        added = 0
        for chunk in noteheads:
            if added >= 256 or chunk.stem is None:
                continue
            x, y = chunk.notehead.center
            # An accepted/split clump must not be added a second time.
            if any(abs(note.center[0] - x) <= chunk.notehead.size[0] / 2 + 1 and
                   abs(note.center[1] - y) <= chunk.notehead.size[1] / 2 + 1
                   for note in notes):
                continue
            staff = nearest_ledger_staff(staffs, x, y)
            if staff is None:
                continue
            tolerance = staff._y_tolerance
            try:
                staff._y_tolerance = 8 * staff.average_unit_size
                extra = original([staff], [chunk], symbols, notehead_pred)
            finally:
                staff._y_tolerance = tolerance
            for note in extra:
                note._omr_extended_ledger = True
                note._omr_staff_space = float(staff.average_unit_size)
            notes.extend(extra)
            added += len(extra)
        return notes

    add_with_ledger_heads._omr_extended_ledger = True
    main.add_notes_to_staffs = add_with_ledger_heads
