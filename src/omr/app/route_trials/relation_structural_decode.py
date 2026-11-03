"""Bounded image-certified bar ownership, long-stem clocks and local glyphs."""
from .relation_measure_realign import fuse as align_bars
from .relation_meter_decode import fuse as decode_clocks
from .relation_local_glyphs import fuse as decode_glyphs


def fuse(root,relation,graph,photo,geometry):
    bars,bar_diagnostics=align_bars(root,relation,graph,photo,geometry)
    clocks,clock_diagnostics=decode_clocks(root,graph,photo,geometry)
    glyphs,glyph_diagnostics=decode_glyphs(root,graph,photo,geometry)
    return bars+clocks+glyphs,{'bars':bar_diagnostics,'clocks':clock_diagnostics,'glyphs':glyph_diagnostics,'reference_supplied':False,'additional_models':False,'additional_PL':False}
