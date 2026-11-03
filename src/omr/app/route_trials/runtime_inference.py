"""CPU inference shared by the development Torch and deployable ONNX routes.

Importing this module does not import Torch, ONNX Runtime or any scorer.
"""
import importlib
import json
from pathlib import Path
import numpy as np
from .relation_session_reuse import session_for


class _LazyTorch:
    def __getattr__(self,name):
        return getattr(importlib.import_module('torch'),name)


torch=_LazyTorch()


def probabilities(model,inputs):
    inputs=np.asarray(inputs,dtype='float32')
    if hasattr(model,'predict_logits'):
        z=model.predict_logits(inputs)
        return 1/(1+np.exp(-np.clip(z,-80,80)))
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(inputs))).numpy()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    compact=path.name.endswith(('.graph.json','.staff.json'))
    options={'indent':None,'separators':(',',':')} if compact else {'indent':2}
    path.write_text(json.dumps(data,ensure_ascii=False,**options)+'\n',encoding='utf8')


class OnnxModel:
    def __init__(self,path):
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
        opts.enable_cpu_mem_arena=False
        self.session=session_for(path,ort.InferenceSession,opts,['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name

    def predict_logits(self,inputs):
        return self.session.run(None,{self.input:np.ascontiguousarray(inputs,dtype='float32')})[0]
