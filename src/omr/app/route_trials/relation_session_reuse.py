"""Explicit page-scoped reuse of four frozen, stateless CPU ONNX sessions.

No output is cached. Outside the context the original constructor is called
unchanged. Every hit compares the complete model bytes and validates the
underlying session/provider/options/input/output state. Unknown assets,
changed bytes, profiling/options and incompatible state use fresh sessions.
Only this registry's references are released at scope exit; caller-owned
OnnxModel objects retain their ordinary lifetime.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import copy
import hashlib
import inspect
from pathlib import Path
import threading
import time

ACTIVE = ContextVar('relation_cpu_session_registry', default=None)
MAX_SESSIONS = 4
MAX_MODEL_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 64


@dataclass(frozen=True)
class Contract:
    sha256: str
    size: int
    input_shape: tuple
    output_shape: tuple


# These exact inline-weight graphs were inspected offline: Conv/Relu/MaxPool/
# Resize/shape operators, or MatMul/Add/Relu; no stochastic/training/custom op.
# A new checkpoint needs new validation; a caller-supplied manifest alone
# never grants an unknown graph session-reuse eligibility.
SUPPORTED_MODELS = {
    'symbol.onnx': Contract('e014e4104d674d699ebc30fae73cab4cf6219fbd7837779932b8839f960b188d',245904,(1,1,160,320),(1,12,80,160)),
    'stroke.onnx': Contract('6e51c6c3bf463181b351a575612902a05b5113ef0fcc9e7baf47d012dacf0639',245592,(1,1,160,320),(1,6,80,160)),
    'edge.onnx': Contract('e2954ea0309d7a72240be1b4ab345d014aab16c4ea6b78bc3203df3b6d0a0c0f',2821,(8,),(1,)),
    'grace.onnx': Contract('8fc63b933026f02fa999d78ed2442de39356d4d3cd22f73d6ac92383ebba4681',39210,('batch',1,32,32),('batch',1)),
}
OPTION_FIELDS = ('intra_op_num_threads','inter_op_num_threads','enable_cpu_mem_arena',
    'enable_mem_pattern','execution_mode','graph_optimization_level','enable_profiling',
    'profile_file_prefix','optimized_model_filepath','log_severity_level','log_verbosity_level',
    'logid','use_deterministic_compute')


def _options(options):
    return tuple((name,str(getattr(options,name,None))) for name in OPTION_FIELDS)


def _signature(values):
    return tuple((value.name,value.type,tuple(value.shape)) for value in values)


def _session_state(session):
    providers=tuple(session.get_providers())
    provider_options=tuple((name,tuple(sorted(value.items())))
                           for name,value in sorted(session.get_provider_options().items()))
    return (id(session._sess),providers,provider_options,_options(session.get_session_options()),
            _signature(session.get_inputs()),_signature(session.get_outputs()))


def _expected_metadata(session,contract):
    return (_signature(session.get_inputs())==(('input','tensor(float)',contract.input_shape),)
            and _signature(session.get_outputs())==(('output','tensor(float)',contract.output_shape),)
            and tuple(session.get_providers())==('CPUExecutionProvider',))


@dataclass
class _Entry:
    payload: bytes
    options: tuple
    session: object
    state: tuple
    number: int


class SessionRegistry:
    def __init__(self,models,validated_manifest,*,page_id,enabled=True):
        if not page_id:raise ValueError('Explicit page identity required')
        self.directory=Path(models).resolve();self.page_id=str(page_id);self.enabled=bool(enabled)
        self.thread=threading.get_ident();self.entries={};self.closed=False;self._serial=0
        self._lock=threading.RLock()
        self._manifest=copy.deepcopy(validated_manifest)
        self.expected={}
        for row in self._manifest.get('files',[]):
            name=row.get('path');contract=SUPPORTED_MODELS.get(name)
            if contract is not None and row.get('sha256')==contract.sha256 and row.get('bytes')==contract.size:
                self.expected[(self.directory/name).resolve()]=contract
        self.stats=dict(enabled=self.enabled,page_id=self.page_id,requests=0,hits=0,misses=0,
            actual_session_creations=0,reuse_saved_session_initializations=0,
            fallback_disabled=0,fallback_unknown_asset=0,fallback_changed_model_bytes=0,
            fallback_incompatible_options=0,fallback_state_or_shape=0,fallback_capacity=0,
            fallback_cross_thread_or_closed_scope=0,model_byte_reads=0,model_bytes_read=0,
            validated_manifest_hits=0,
            model_verification_s=0.,session_creation_s=0.,lookup_s=0.,
            peak_cached_sessions=0,peak_cached_model_bytes=0,live_cached_sessions=0,
            live_cached_model_bytes=0,live_sessions_before_cleanup=None,scope_cleanup_verified=False,
            maximum_sessions=MAX_SESSIONS,maximum_cached_model_payload_bytes=MAX_MODEL_BYTES,
            native_session_memory_is_not_model_payload_bytes=True,model_outputs_cached=False,
            cross_page_reuse=False,maximum_events=MAX_EVENTS,dropped_events=0,events=[])

    def _event(self,event):
        if len(self.stats['events'])<MAX_EVENTS:self.stats['events'].append(event)
        else:self.stats['dropped_events']+=1

    def _fresh(self,path,constructor,options,providers,reason,*,payload=None):
        self.stats['misses']+=1;self.stats['actual_session_creations']+=1
        if reason:self.stats['fallback_'+reason]+=1
        tick=time.perf_counter()
        try:
            session=constructor(str(path) if payload is None else payload,sess_options=options,providers=providers)
        except Exception:
            self._event(dict(model=Path(path).name,action='creation_failed',reason=reason))
            raise
        finally:self.stats['session_creation_s']+=time.perf_counter()-tick
        return session

    def _resident(self):
        self.stats['live_cached_sessions']=len(self.entries)
        self.stats['live_cached_model_bytes']=sum(len(entry.payload) for entry in self.entries.values())
        self.stats['peak_cached_sessions']=max(self.stats['peak_cached_sessions'],len(self.entries))
        self.stats['peak_cached_model_bytes']=max(self.stats['peak_cached_model_bytes'],self.stats['live_cached_model_bytes'])

    def obtain(self,path,constructor,options,providers):
        with self._lock:
            self.stats['requests']+=1;resolved=Path(path).resolve();contract=self.expected.get(resolved)
            reason=None
            if self.closed or threading.get_ident()!=self.thread:reason='cross_thread_or_closed_scope'
            elif not self.enabled:reason='disabled'
            elif contract is None:reason='unknown_asset'
            elif (tuple(providers)!=('CPUExecutionProvider',)
                  or options.intra_op_num_threads!=4 or options.inter_op_num_threads!=1
                  or options.enable_cpu_mem_arena is not False or getattr(options,'enable_profiling',False)
                  or getattr(options,'optimized_model_filepath','')):reason='incompatible_options'
            if reason:
                session=self._fresh(path,constructor,options,providers,reason)
                self._event(dict(model=Path(path).name,action='fresh',reason=reason))
                return session
            tick=time.perf_counter()
            try:
                with resolved.open('rb') as source:data=source.read(contract.size+1)
            except OSError:
                self.stats['model_verification_s']+=time.perf_counter()-tick
                # Original constructor retains ownership of missing/unreadable
                # file error behavior; do not return an old cached session.
                session=self._fresh(path,constructor,options,providers,'changed_model_bytes')
                self._event(dict(model=resolved.name,action='fresh',reason='unreadable_model'))
                return session
            digest=hashlib.sha256(data).hexdigest();self.stats['model_byte_reads']+=1
            self.stats['model_bytes_read']+=len(data);self.stats['model_verification_s']+=time.perf_counter()-tick
            if digest!=contract.sha256 or len(data)!=contract.size:
                self.entries.pop(resolved,None);self._resident()
                session=self._fresh(path,constructor,options,providers,'changed_model_bytes')
                self._event(dict(model=resolved.name,action='fresh',reason='changed_model_bytes',observed_sha256=digest))
                return session
            tick=time.perf_counter();key=_options(options);entry=self.entries.get(resolved)
            state_valid=False
            if entry is not None:
                try:state_valid=(entry.payload==data and entry.options==key and entry.state==_session_state(entry.session)
                                 and _expected_metadata(entry.session,contract))
                except (AttributeError,TypeError,ValueError):state_valid=False
            self.stats['lookup_s']+=time.perf_counter()-tick
            if state_valid:
                self.stats['hits']+=1;self.stats['reuse_saved_session_initializations']+=1
                self._event(dict(model=resolved.name,action='reuse',session_number=entry.number,
                    model_sha256=digest,model_bytes=len(data),metadata_and_state_verified=True))
                return entry.session
            if entry is not None:
                del self.entries[resolved];self.stats['fallback_state_or_shape']+=1;self._resident()
            session=self._fresh(path,constructor,options,providers,None,payload=data)
            try:cacheable=_expected_metadata(session,contract) and _options(session.get_session_options())==key
            except (AttributeError,TypeError,ValueError):cacheable=False
            if not cacheable:self.stats['fallback_state_or_shape']+=1
            elif len(self.entries)>=MAX_SESSIONS or self.stats['live_cached_model_bytes']+len(data)>MAX_MODEL_BYTES:
                cacheable=False;self.stats['fallback_capacity']+=1
            if cacheable:
                self._serial+=1
                self.entries[resolved]=_Entry(data,key,session,_session_state(session),self._serial)
            self._resident()
            self._event(dict(model=resolved.name,action='created',cached=cacheable,
                model_sha256=digest,model_bytes=len(data),session_number=self._serial if cacheable else None))
            return session

    def statistics(self):
        return dict(self.stats,events=[dict(event) for event in self.stats['events']])

    def close(self):
        with self._lock:
            self.stats['live_sessions_before_cleanup']=len(self.entries)
            self.entries.clear();self.closed=True;self._resident()
            self.stats['scope_cleanup_verified']=not self.entries


def session_for(path,constructor,options,providers):
    active=ACTIVE.get()
    if active is None:
        return constructor(str(path),sess_options=options,providers=providers)
    return active.obtain(path,constructor,options,providers)


def validated_manifest_for(models):
    """Private copy of the same-directory scope's entry validation result.

    No file read/hash is repeated here. Different/nested pages, directories,
    closed scopes and other threads never borrow another validation result.
    Returning a small deep copy prevents callers mutating the stored proof.
    """
    active=ACTIVE.get()
    if (active is None or active.closed or threading.get_ident()!=active.thread
            or Path(models).resolve()!=active.directory):
        return None
    with active._lock:
        active.stats['validated_manifest_hits']+=1
        return copy.deepcopy(active._manifest)


@contextmanager
def cpu_session_scope(models,validated_manifest,*,page_id,enabled=True):
    """Independent registry for this explicit page; nested scopes never share."""
    registry=SessionRegistry(models,validated_manifest,page_id=page_id,enabled=enabled)
    token=ACTIVE.set(registry)
    try:yield registry
    finally:ACTIVE.reset(token);registry.close()


def scoped_cpu_sessions(validator,*,enabled=True):
    """Optional root-owned integration around an enhance-like callable.

    Validator must return the already verified manifest. The wrapper never
    edits PageInputs or writes files; telemetry is attached AFTER cleanup.
    Baseline profiling uses enabled=False for real constructor call counts.
    """
    def decorate(function):
        signature=inspect.signature(function)
        @wraps(function)
        def run(*args,**kwargs):
            bound=signature.bind(*args,**kwargs);models=bound.arguments['models'];page=bound.arguments['photo']
            manifest=validator(models)
            with cpu_session_scope(models,manifest,page_id=str(Path(page).resolve()),enabled=enabled) as registry:
                result=function(*args,**kwargs)
            result['onnx_session_reuse']=registry.statistics()
            return result
        return run
    return decorate
