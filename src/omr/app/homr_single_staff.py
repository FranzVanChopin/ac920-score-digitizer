"""Preserve decoder symbols for an image that contains one five-line staff.

Homr normally discards all symbols labelled ``lower`` for a single staff.
Dense bass passages can be correctly decoded with that label. When the
detected image has exactly five lines and the decoded clef is unambiguous,
both labels refer to the sole visible staff. Normalize them without another
model inference. A possible two-clef interpretation keeps upstream behavior.
"""
from types import SimpleNamespace


def normalize_single_staff(symbols, staff):
    if staff.is_grandstaff:
        return symbols, 0
    if not staff.grid or any(len(p.y)!=5 for p in staff.grid):
        return [s for s in symbols if s.position!='lower'], 0
    clefs={s.rhythm for s in symbols if s.rhythm.startswith('clef_')}
    if len(clefs)!=1:
        return [s for s in symbols if s.position!='lower'], 0
    count=sum(s.position=='lower' for s in symbols)
    return [s.to_upper_position() for s in symbols], count


def install_single_staff_normalization():
    import homr.staff_parsing_tromr as parser
    original=parser.predict_best
    records=[]
    def predict(image,staff,config):
        if staff.is_grandstaff:
            return original(image,staff,config)
        raw=original(image,SimpleNamespace(is_grandstaff=True),config)
        result,count=normalize_single_staff(raw,staff)
        if count:
            records.append({'method':'sole_five_line_staff','preserved_lower_tokens':count,
                            'decoded_clefs':sorted({s.rhythm for s in raw if s.rhythm.startswith('clef_')})})
        return result
    parser.predict_best=predict
    return records
