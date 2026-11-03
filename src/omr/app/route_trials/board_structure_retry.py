"""One optional sequential hypothesis; fuse certified local structure only.

The primary XML and geometry remain live until every guard and sidecar update
has succeeded. Retry/model/parser failures return those original objects.
"""
import copy
import json
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as E
import cv2
import numpy as np
from homr_staff_pair_recovery import find_pair_candidates
from .relation_left_crop_trial import candidate_systems


def candidates(geometry,image):
    return dict(left=candidate_systems(geometry,image),pairs=find_pair_candidates(geometry,image))


def recover(root,geometry,image,photo,staged,command,env,models,relations,staff_scale,initial_meter,postprocess_fn,enhance_fn,proof=None):
    proof=candidates(geometry,image) if proof is None else proof
    report=dict(enabled=True,retried=False,changes=[],maximum_alternate_passes=1,candidates=proof)
    if not proof['left'] and not proof['pairs']:return root,geometry,report
    if command is not None and '--recover-structure' in command:
        report['reason']='The primary pass already used the same image-certified structure recovery; no duplicate model pass'
        return root,geometry,report
    if proof['left'] and proof['pairs']:
        report['reason']='Competing system-count changes require separate evidence';return root,geometry,report
    try:
        parent=Path(staged).parent/'structure_retry';parent.mkdir(exist_ok=True)
        target=parent/Path(staged).name;shutil.copyfile(staged,target)
        cmd=list(command);cmd[2]=str(target);cmd.append('--recover-structure');report['retried']=True
        child=subprocess.run(cmd,capture_output=True,text=True,env=env,check=False)
        if child.returncode:raise ValueError('Alternate Homr failed: '+child.stderr[-1000:])
        xml=target.with_suffix('.musicxml');gp=target.with_suffix('.geometry.json')
        candidate=E.parse(xml).getroot();other=json.loads(gp.read_text(encoding='utf8'))
        if candidate.tag!='score-partwise' or len(candidate.findall('part'))!=1 or not candidate.findall('part/measure'):
            raise ValueError('Alternate score structure invalid')
        raster=cv2.imread(str(target.with_suffix('.preprocessed.png')),0)
        if raster is None or raster.shape!=image.shape or not np.array_equal(raster,image):
            raise ValueError('Retry preprocessing differs; cannot fuse coordinates')
        postprocess_fn(candidate,other,raster,initial_meter)
        if relations=='off' or not (Path(models)/'manifest.json').is_file():
            raise ValueError('Validated relation assets are required for structure recovery')
        E.ElementTree(candidate).write(xml,encoding='utf8',xml_declaration=True)
        output=parent/'enhanced.musicxml';opts={'supplementary_staff_scale':16.} if staff_scale else {}
        enhanced=enhance_fn(photo,gp,xml,models,output,**opts)
        if enhanced.get('fallback'):raise ValueError('Alternate relation enhancement fell back')
        candidate=E.parse(output).getroot();trial=copy.deepcopy(root)
        if proof['left']:
            from .testset3_measure_recovery import fuse
        else:
            from .relation_staff_pair_conservative import fuse
        changes=fuse(trial,candidate,geometry,other,image)
        if proof['left'] and not changes:
            # Keep the narrower previously validated sharp-key recovery.
            from .relation_left_crop_trial import fuse as legacy_fuse
            trial=copy.deepcopy(root)
            changes=legacy_fuse(trial,candidate,geometry,other,image)
        if not changes:return root,geometry,report
        result=copy.deepcopy(geometry)
        if proof['left']:
            for change in changes:
                index=change['system']-1
                result['systems'][index]=copy.deepcopy(other['systems'][index])
        else:
            from .relation_staff_pair_conservative import fusion_geometry
            result=fusion_geometry(geometry,other,changes)
        result['local_structure_recovery']=changes
        report.update(changes=changes,alternate_relation_enabled=True,unaffected_systems_preserved=True)
        return trial,result,report
    except Exception as exc:
        report.update(fallback=True,error=str(exc));return root,geometry,report
