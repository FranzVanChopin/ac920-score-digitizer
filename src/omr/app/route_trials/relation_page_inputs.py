"""Scoped, bounded reuse of immutable relation inputs; never reuse score nodes.

Scaled photos remain private writable arrays for every caller. Geometry is
shared only inside the synchronous enhancement and checked before accepting
its result. Mutable MusicXML is reparsed from immutable bytes for each reader.
"""
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as E

import cv2

ACTIVE = ContextVar('relation_page_inputs', default=None)
MAX_IMAGE_BYTES = 32_000_000
MAX_FILE_BYTES = 16_000_000


def stamp(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


class PageInputs:
    def __init__(self, photo):
        self.photo = Path(photo).resolve()
        self.photo_stamp = stamp(self.photo)
        self.images = {}
        self.geometry = {}
        self.xml = OrderedDict()
        self.stats = dict(scaled_photo_hits=0, scaled_photo_misses=0,
                          scaled_image_resident_bytes=0, scaled_image_copy_bytes=0,
                          geometry_hits=0, geometry_misses=0, geometry_guard_verified=False,
                          geometry_encoded_bytes=0, xml_byte_hits=0, xml_byte_misses=0,
                          xml_resident_bytes=0, xml_parse_s=0., hash_hits=0, hash_misses=0,
                          private_writable_images=True, private_score_nodes=True,
                          maximum_image_bytes=MAX_IMAGE_BYTES, maximum_file_bytes=MAX_FILE_BYTES)
        self.photo_sha = None

    def check_photo(self):
        if stamp(self.photo) != self.photo_stamp:
            raise ValueError('Inference photograph changed during relation enhancement')

    def digest_photo(self, photo):
        path = Path(photo).resolve()
        if path != self.photo:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        self.check_photo()
        if self.photo_sha is None:
            self.photo_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.stats['hash_misses'] += 1
        else:
            self.stats['hash_hits'] += 1
        return self.photo_sha

    def scaled(self, photo, fx, fy, interpolation):
        path = Path(photo).resolve()
        if path != self.photo:
            return _scaled(path, fx, fy, interpolation)
        self.check_photo()
        key = fx, fy, interpolation
        if key not in self.images:
            self.stats['scaled_photo_misses'] += 1
            value = _scaled(path, fx, fy, interpolation)
            if value is None or self.stats['scaled_image_resident_bytes'] + value.nbytes > MAX_IMAGE_BYTES:
                return value
            value.setflags(write=False)
            self.images[key] = value
            self.stats['scaled_image_resident_bytes'] += value.nbytes
        else:
            self.stats['scaled_photo_hits'] += 1
        result = self.images[key].copy()
        self.stats['scaled_image_copy_bytes'] += result.nbytes
        return result

    def geometry_json(self, filename):
        path = Path(filename).resolve()
        if path in self.geometry:
            value, identity, expected = self.geometry[path]
            if stamp(path) != identity:
                raise ValueError('Relation geometry file changed during enhancement')
            self.stats['geometry_hits'] += 1
            return value
        data = path.read_bytes()
        value = json.loads(data.decode('utf-8'))
        if self.stats['geometry_encoded_bytes'] + len(data) <= MAX_FILE_BYTES:
            self.geometry[path] = value, stamp(path), fingerprint(value)
            self.stats['geometry_encoded_bytes'] += len(data)
            self.stats['geometry_misses'] += 1
        return value

    def score(self, filename):
        path = Path(filename).resolve()
        identity = stamp(path)
        entry = self.xml.get(path)
        if entry is not None and entry[0] == identity:
            data = entry[1]
            self.xml.move_to_end(path)
            self.stats['xml_byte_hits'] += 1
        else:
            data = path.read_bytes()
            self.stats['xml_byte_misses'] += 1
            if entry is not None:
                self.stats['xml_resident_bytes'] -= len(entry[1])
                del self.xml[path]
            if len(data) <= MAX_FILE_BYTES:
                while self.xml and self.stats['xml_resident_bytes'] + len(data) > MAX_FILE_BYTES:
                    _, old = self.xml.popitem(last=False)
                    self.stats['xml_resident_bytes'] -= len(old[1])
                self.xml[path] = identity, data
                self.stats['xml_resident_bytes'] += len(data)
        start = perf_counter()
        result = E.ElementTree(E.fromstring(data))
        self.stats['xml_parse_s'] += perf_counter() - start
        return result

    def verify(self):
        self.check_photo()
        for path, (value, identity, expected) in self.geometry.items():
            if stamp(path) != identity or fingerprint(value) != expected:
                raise ValueError('Shared read-only relation geometry was modified')
        self.stats['geometry_guard_verified'] = True

    def clear(self):
        self.images.clear()
        self.geometry.clear()
        self.xml.clear()


def _scaled(photo, fx, fy, interpolation):
    raw = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.size > 16_000_000 or max(raw.shape) > 8192:
        return None
    return cv2.resize(raw, None, fx=fx, fy=fy, interpolation=interpolation)


def photo_digest(photo):
    active = ACTIVE.get()
    return active.digest_photo(photo) if active else hashlib.sha256(Path(photo).read_bytes()).hexdigest()


def scaled_photo(photo, fx, fy, interpolation=cv2.INTER_AREA):
    active = ACTIVE.get()
    return active.scaled(photo, fx, fy, interpolation) if active else _scaled(photo, fx, fy, interpolation)


def geometry_json(filename):
    active = ACTIVE.get()
    return active.geometry_json(filename) if active else json.loads(Path(filename).read_text(encoding='utf-8'))


def parse_score(filename):
    active = ACTIVE.get()
    return active.score(filename) if active else E.parse(filename)


def cached_relation_inputs(function):
    @wraps(function)
    def run(photo, *args, **kwargs):
        context = PageInputs(photo)
        token = ACTIVE.set(context)
        try:
            from .relation_geometry_index import geometry_index
            def copy_statistics():
                module=sys.modules.get('route_trials.relation_native_head_state_lazy')
                return module.statistics() if module is not None else dict(calls=0,legacy_fallbacks=0,noop_borrowed_graphs=0,confirmed_full_graph_copies=0)
            copy_before=copy_statistics()
            with geometry_index() as index:
                result = function(photo, *args, **kwargs)
                context.verify()
                result['page_input_reuse'] = dict(context.stats)
                result['geometry_index'] = index.statistics()
                result['native_graph_copy']={key:value-copy_before[key] for key,value in copy_statistics().items()}
                return result
        finally:
            ACTIVE.reset(token)
            context.clear()
    return run
