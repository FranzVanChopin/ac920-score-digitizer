"""Exact bounded integer note/head alignment; None selects the legacy path.

Only the DP evaluation order changes. Forward/backward gap costs and the
unique-match/skip test remain identical, and returned keys and values retain
the caller's mutable XML/head identities. No input or result is cached.
"""
import xml.etree.ElementTree as ET

import numpy as np


def _forward(cost):
    n, m = cost.shape
    inf = 3 * (n + m) + 1
    table = np.full((n + 1, m + 1), inf, np.int32)
    table[:, 0] = 3 * np.arange(n + 1, dtype=np.int32)
    table[0, :] = 3 * np.arange(m + 1, dtype=np.int32)
    for diagonal in range(2, n + m + 1):
        i = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1, dtype=np.intp)
        j = diagonal - i
        table[i, j] = np.minimum(
            np.minimum(table[i - 1, j] + 3, table[i, j - 1] + 3),
            table[i - 1, j - 1] + cost[i - 1, j - 1],
        )
    return table


def try_align(notes, heads, position):
    """Return the exact mapping for safe large inputs, otherwise None."""
    if type(notes) is not list or type(heads) is not list:
        return None
    n, m = len(notes), len(heads)
    if not n or not m or max(n, m) > 256 or n * m < 1024:
        return None
    if any(type(note) is not ET.Element for note in notes):
        return None
    if any(type(head) is not dict or type(head.get('position')) is not int
           or type(head.get('extended_ledger', False)) not in (bool, type(None))
           for head in heads):
        return None
    try:
        positions = [position(note)[1] for note in notes]
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError):
        return None
    if (any(type(value) is not int or abs(value) > 1024 for value in positions)
            or any(abs(head['position']) > 1024 for head in heads)):
        return None
    note_positions = np.asarray(positions, np.int32)
    head_positions = np.asarray([head['position'] for head in heads], np.int32)
    delta = abs(note_positions[:, None] - head_positions[None, :])
    extension = np.asarray([bool(head.get('extended_ledger')) for head in heads])
    inf = 3 * (n + m) + 1
    cost = np.where(
        delta == 0, 0,
        np.where(extension[None, :] & (note_positions[:, None] >= 16) & (delta <= 2), delta, inf),
    ).astype(np.int32)
    forward = _forward(cost)
    backward = _forward(cost[::-1, ::-1])[::-1, ::-1]
    best = forward[n, m]
    mapped = {}
    for i, note in enumerate(notes):
        if np.any(forward[i, :] + 3 + backward[i + 1, :] == best):
            continue
        possible = np.flatnonzero(forward[i, :m] + cost[i, :] + backward[i + 1, 1:] == best)
        if len(possible) == 1:
            mapped[note] = heads[int(possible[0])]
    return mapped
