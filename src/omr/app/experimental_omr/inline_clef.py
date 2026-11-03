"""Conservative detection of a bass clef inserted inside a staff system.

The two small dots to the right of an F clef distinguish it from noteheads,
rests, and the similar-looking left half of a beamed note group.  Work on
staff-line-removed ink so a vertical barline cannot join either dot.
"""

from __future__ import annotations

import cv2
import numpy as np

from .staff import StaffGeometry


def find_inline_bass_clefs(ink: np.ndarray,
                           staff: StaffGeometry) -> tuple[float, ...]:
    """Return x positions immediately after visually supported F clefs.

    The staff's opening clef is handled by the ordinary clef detector, so
    this search begins eight staff spaces inside the music region.
    """
    space = float(staff.space)
    x0 = max(0, int(round(staff.x0 + 8.0 * space)))
    x1 = min(ink.shape[1], int(round(staff.x1)))
    y0 = max(0, int(round(staff.y_top_line - 0.35 * space)))
    y1 = min(ink.shape[0], int(round(staff.y_bottom_line + 0.55 * space)))
    if x1 - x0 < 3 * space or y1 <= y0:
        return ()

    region = ink[y0:y1, x0:x1].astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(region, 8)
    components = []
    for i in range(1, count):
        rx, ry, width, height, area = map(float, stats[i])
        components.append((rx + x0, ry + y0, width, height, area,
                           float(centroids[i, 0]) + x0,
                           float(centroids[i, 1]) + y0))

    # Bass-clef dots are small, disconnected round components, centred on
    # either side of the second staff line from the top.
    dots = [c for c in components
            if 0.23 * space <= c[2] <= 0.62 * space
            and 0.23 * space <= c[3] <= 0.62 * space
            and 0.07 * space * space <= c[4] <= 0.29 * space * space]
    top_dots = [c for c in dots if 0.32 * space <= c[6] - staff.y_top_line
                <= 0.95 * space]
    lower_dots = [c for c in dots if 1.08 * space <= c[6] - staff.y_top_line
                  <= 1.72 * space]
    found = []
    for upper in top_dots:
        for lower in lower_dots:
            if abs(upper[5] - lower[5]) > 0.23 * space:
                continue
            if not 0.60 * space <= lower[6] - upper[6] <= 1.13 * space:
                continue
            dot_left = min(upper[0], lower[0])
            bodies = [c for c in components
                      if 1.1 * space <= c[2] <= 2.5 * space
                      and 2.15 * space <= c[3] <= 3.7 * space
                      and c[4] >= 0.7 * space * space
                      and -0.25 * space <= c[1] - staff.y_top_line
                      <= 0.5 * space
                      and 0 <= dot_left - (c[0] + c[2]) <= 0.45 * space]
            if bodies:
                found.append(max(upper[0] + upper[2],
                                 lower[0] + lower[2]) + 0.2 * space)

    found.sort()
    return tuple(x for i, x in enumerate(found)
                 if i == 0 or x - found[i - 1] > 1.0 * space)
