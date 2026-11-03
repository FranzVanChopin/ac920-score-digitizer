"""Persist full decision proofs once before serializing final page reports."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def persist_edge_traces(report,output):
    """Replace embedded proofs with verified persistent, atomic sidecars.

    Call only after recognition/repairs have finished. This changes diagnostic
    storage, never the graph or MusicXML. JSON field/list order is retained.
    """
    output=Path(output);saved=[]
    def visit(value):
        if isinstance(value,dict):
            for key,child in list(value.items()):
                if key=='classical_edge_gate_trace' and isinstance(child,dict):
                    if child.get('storage')=='sidecar':continue
                    data=(json.dumps(child,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf8')
                    digest=hashlib.sha256(data).hexdigest()
                    # A failed replacement of an existing score must not
                    # overwrite the proof referenced by the previous score.
                    path=output.with_suffix(f'.edge_gate.{digest}.json')
                    descriptor,temporary=tempfile.mkstemp(prefix='.omr-edge-',suffix='.json',dir=output.parent)
                    try:
                        with os.fdopen(descriptor,'wb') as sink:
                            sink.write(data);sink.flush();os.fsync(sink.fileno())
                        os.replace(temporary,path)
                    finally:
                        if os.path.exists(temporary):os.unlink(temporary)
                    entry=dict(storage='sidecar',path=str(path.resolve()),
                               sha256=digest,bytes=len(data),
                               complete_decision_trace=True)
                    value[key]=entry;saved.append(entry)
                else:visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(report)
    return saved
