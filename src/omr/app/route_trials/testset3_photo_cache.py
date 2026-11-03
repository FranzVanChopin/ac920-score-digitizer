"""Bounded exact photo decode reuse within one synchronous enhancement.

Only the immutable inference photograph is cached; each consumer receives a
private copy. No generated score/geometry, model output or reference is cached.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import time
from unittest.mock import patch

import cv2

MAXIMUM_CACHE_BYTES = 64_000_000


@contextmanager
def photo_reads(photo):
    source = Path(photo).resolve(strict=True)
    identity = source.stat()
    original = cv2.imread
    cache = {}
    stats = dict(hits=0, misses=0, resident_bytes=0, decode_s=0., copy_s=0.,
                 private_consumer_copies=True, maximum_bytes=MAXIMUM_CACHE_BYTES,
                 scores_geometry_or_references_cached=False)
    def read(filename, flags=cv2.IMREAD_COLOR):
        if flags not in (cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR) or Path(filename).resolve() != source:
            return original(filename, flags)
        now = source.stat()
        if (now.st_size, now.st_mtime_ns) != (identity.st_size, identity.st_mtime_ns):
            raise ValueError('Inference photograph changed within enhancement')
        if flags not in cache:
            started = time.perf_counter()
            value = original(filename, flags)
            stats['decode_s'] += time.perf_counter()-started
            stats['misses'] += 1
            if value is None or value.nbytes+stats['resident_bytes'] > MAXIMUM_CACHE_BYTES:
                return value
            value.setflags(write=False)
            cache[flags] = value
            stats['resident_bytes'] += value.nbytes
        else:
            stats['hits'] += 1
        started = time.perf_counter()
        result = cache[flags].copy()
        stats['copy_s'] += time.perf_counter()-started
        return result
    try:
        with patch.object(cv2, 'imread', read):
            yield stats
    finally:
        cache.clear()


def cached_photo_reads(function):
    @wraps(function)
    def run(photo, *args, **kwargs):
        with photo_reads(photo) as stats:
            result = function(photo, *args, **kwargs)
        result['photo_decode_cache'] = stats
        return result
    return run
