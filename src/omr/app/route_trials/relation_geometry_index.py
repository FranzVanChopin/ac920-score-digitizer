"""Scoped, mutation-checked copies of ordinary float staff polylines.

No cache is active by default. Each hit verifies shape, scalar types, values
and signed zero before reusing a private read-only ndarray. The source is
strongly retained, so identities cannot be recycled inside a scope. Generic
containers/numbers and nonfinite coordinates use the original np.array path.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import marshal
import threading
import numpy as np

_active=ContextVar('relation_geometry_index',default=None)
_SCALAR_TYPES=frozenset((int,float,np.float16,np.float32,np.float64,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64))
MAXIMUM_LINES=512
MAXIMUM_POINTS=131_072
MAXIMUM_LINE_POINTS=4096


@dataclass
class _Entry:
    source: list
    snapshot: list
    array: np.ndarray
    zeros: list
    signature: bytes


def _ordinary(polyline):
    if type(polyline) is not list or not 0<len(polyline)<=MAXIMUM_LINE_POINTS:return False
    return all(type(p) is list and len(p)==2 and type(p[0]) is float and type(p[1]) is float
               and math.isfinite(p[0]) and math.isfinite(p[1]) for p in polyline)


def _unchanged(polyline,entry):
    # marshal's C encoder preserves container/scalar types and float bits,
    # including signed zero. It rejects custom objects without invoking their
    # comparison or conversion hooks. Version 2 has no reference-table flags,
    # so an unrelated reference to a float cannot change the signature.
    if entry.source is not polyline or len(polyline)!=len(entry.snapshot):return False
    try:return marshal.dumps(polyline,2)==entry.signature
    except (ValueError,TypeError,RecursionError):return False


class GeometryIndex:
    def __init__(self,maximum_lines=MAXIMUM_LINES,maximum_points=MAXIMUM_POINTS):
        if not isinstance(maximum_lines,int) or not 1<=maximum_lines<=MAXIMUM_LINES:raise ValueError('Invalid line cache budget')
        if not isinstance(maximum_points,int) or not 1<=maximum_points<=MAXIMUM_POINTS:raise ValueError('Invalid coordinate cache budget')
        self.maximum_lines=maximum_lines;self.maximum_points=maximum_points
        self.owner_thread=threading.get_ident();self.entries=OrderedDict();self.points=0
        self.counts=dict(hits=0,misses=0,mutation_invalidations=0,fallbacks=0,evictions=0,peak_lines=0,peak_points=0)

    def array(self,polyline):
        if threading.get_ident()!=self.owner_thread:
            self.counts['fallbacks']+=1;return np.array(polyline)
        identity=id(polyline);entry=self.entries.get(identity)
        if entry is not None:
            if _unchanged(polyline,entry):
                self.counts['hits']+=1;self.entries.move_to_end(identity);return entry.array
            self.points-=len(entry.snapshot);del self.entries[identity];self.counts['mutation_invalidations']+=1
        if not _ordinary(polyline) or len(polyline)>self.maximum_points:
            self.counts['fallbacks']+=1;return np.array(polyline)
        # Same np.array constructor and dtype inference as the original helper.
        array=np.array(polyline);array.setflags(write=False)
        snapshot=[p.copy() for p in polyline]
        zeros=[(i,j,math.copysign(1.,v)) for i,p in enumerate(polyline) for j,v in enumerate(p) if v==0.]
        while self.entries and (len(self.entries)>=self.maximum_lines or self.points+len(snapshot)>self.maximum_points):
            _,old=self.entries.popitem(last=False);self.points-=len(old.snapshot);self.counts['evictions']+=1
        self.entries[identity]=_Entry(polyline,snapshot,array,zeros,marshal.dumps(polyline,2));self.points+=len(snapshot);self.counts['misses']+=1
        self.counts['peak_lines']=max(self.counts['peak_lines'],len(self.entries));self.counts['peak_points']=max(self.counts['peak_points'],self.points)
        return array

    def statistics(self):
        return dict(self.counts,resident_lines=len(self.entries),resident_points=self.points,
                    maximum_lines=self.maximum_lines,maximum_points=self.maximum_points,
                    input_mutation_checked_on_every_hit=True,arrays_read_only=True)

    def close(self):self.entries.clear();self.points=0


@contextmanager
def geometry_index(*,maximum_lines=MAXIMUM_LINES,maximum_points=MAXIMUM_POINTS):
    index=GeometryIndex(maximum_lines,maximum_points);token=_active.set(index)
    try:yield index
    finally:_active.reset(token);index.close()


def polyline_array(polyline):
    index=_active.get()
    return np.array(polyline) if index is None else index.array(polyline)


def query_index(x,y):
    index=_active.get()
    return index if index is not None and type(x) in _SCALAR_TYPES and type(y) in _SCALAR_TYPES else None


def indexed_array(polyline,index):
    return np.array(polyline) if index is None else index.array(polyline)
