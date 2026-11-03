"""Recover one missed piano sibling from existing segmentation and raster.

For a sparse page with one detected staff only, require five long, equally
spaced missing lines and both outer full-height connections. No new neural
inference is loaded. Ambiguous candidates leave Homr's result unchanged.
"""

from statistics import median


def missing_sibling_grid(staff, mask, image):
    import cv2
    import numpy as np
    if len(staff.grid[0].y) != 5 or staff.max_x-staff.min_x < 12*staff.average_unit_size:
        return None
    space = float(staff.average_unit_size)
    l, r = max(0, round(staff.min_x)), min(mask.shape[1], round(staff.max_x))
    projection = (mask[:, l:r] > 0).mean(axis=1)
    bands = []
    for y in np.flatnonzero(projection >= 0.55):
        if not bands or y-bands[-1][-1] > 1:
            bands.append([])
        bands[-1].append(int(y))
    centers = [median(b) for b in bands]
    candidates = []
    for i in range(len(centers)-4):
        lines = centers[i:i+5]
        if (not 5*space < lines[0]-staff.max_y < 25*space
                or any(abs(b-a-space) > 0.2*space for a, b in zip(lines, lines[1:]))):
            continue
        fitted = []
        for center in lines:
            points = []
            for x in range(l, r, 10):
                top, bottom = max(0, round(center-space*.45)), min(mask.shape[0], round(center+space*.45)+1)
                values = (mask[top:bottom, max(l, x-10):min(r, x+11)] > 0).mean(axis=1)
                rows = np.flatnonzero(values >= 0.4)
                if len(rows):
                    points.append((x, top+float(median(rows))))
            if len(points) < 0.7*((r-l)/10):
                break
            x, y = np.array(points).T
            a, b = np.polyfit(x, y, 1)
            residual = np.abs(y-(a*x+b))
            keep = residual <= 1.5
            if keep.sum() < 0.7*len(points):
                break
            fitted.append(np.polyfit(x[keep], y[keep], 1))
        if len(fitted) != 5:
            continue
        grid = [(float(p.x), [float(a*p.x+b) for a, b in fitted], float(p.angle)) for p in staff.grid]
        if any(abs(b-a-space) > 0.2*space for _, ys, _ in grid for a, b in zip(ys, ys[1:])):
            continue
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        top, bottom = round(staff.min_y), round(max(ys[-1] for _, ys, _ in grid))
        if top < 0 or bottom >= image.shape[0]:
            continue
        connected = []
        for edge in (staff.min_x, staff.max_x):
            choices = [x for x in range(max(0, round(edge-space)), min(image.shape[1], round(edge+space)+1))
                       if (image[top:bottom, max(0, x-1):x+2].min(axis=1) < 150).mean() >= .92]
            connected.append(bool(choices))
        if all(connected):
            candidates.append(grid)
    return candidates[0] if len(candidates) == 1 else None


def install_staff_recovery():
    import homr.main as main
    from homr.model import Staff, StaffPoint
    original = main.detect_staff
    if getattr(original, "_omr_staff_recovery", False):
        original._omr_reports.clear()
        return original._omr_reports
    reports = []

    def detect(debug, mask, *args):
        staffs = original(debug, mask, *args)
        if len(staffs) == 1:
            grid = missing_sibling_grid(staffs[0], mask, debug.original_image)
            if grid:
                recovered = Staff([StaffPoint(x, y, angle) for x, y, angle in grid])
                staffs.append(recovered)
                reports.append({"method": "five_long_equal_lines_and_two_outer_connections",
                                "y_range": [float(recovered.min_y), float(recovered.max_y)],
                                "x_range": [float(recovered.min_x), float(recovered.max_x)]})
        return staffs

    detect._omr_staff_recovery = True
    detect._omr_reports = reports
    main.detect_staff = detect
    return reports
