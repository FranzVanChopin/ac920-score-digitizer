"""One sequential alternate recognition; append-only certified tail recovery."""
import copy,json,subprocess,shutil
from pathlib import Path
import xml.etree.ElementTree as E
import cv2,numpy as np
from .relation_tail_splice_experiment import fuse


def recover(root,geometry,image,photo,staged,command,env,models,relations,staff_scale,initial_meter,postprocess_fn,enhance_fn):
    report=dict(enabled=True,retried=False,changes=[],maximum_alternate_passes=1,old_measures_preserved=True)
    if not geometry.get('staff_tail_candidates'):return root,geometry,report
    try:
        parent=Path(staged).parent/'tail_retry';parent.mkdir(exist_ok=True)
        target=parent/Path(staged).name;shutil.copyfile(staged,target)
        cmd=list(command);cmd[2]=str(target);cmd.append('--extend-staff-tail');report['retried']=True
        child=subprocess.run(cmd,capture_output=True,text=True,env=env,check=False)
        if child.returncode:raise ValueError('Alternate Homr failed: '+child.stderr[-1000:])
        xml=target.with_suffix('.musicxml');gp=target.with_suffix('.geometry.json')
        candidate=E.parse(xml).getroot();other=json.loads(gp.read_text(encoding='utf8'))
        raster=cv2.imread(str(target.with_suffix('.preprocessed.png')),0)
        if raster is None or raster.shape!=image.shape or not np.array_equal(raster,image):raise ValueError('Retry preprocessing differs; cannot splice coordinates')
        postprocess_fn(candidate,other,raster,initial_meter)
        if relations!='off' and (Path(models)/'manifest.json').is_file():
            E.ElementTree(candidate).write(xml,encoding='utf8',xml_declaration=True)
            output=parent/'enhanced.musicxml';opts={'supplementary_staff_scale':16.} if staff_scale else {}
            enhanced=enhance_fn(photo,gp,xml,models,output,**opts)
            if enhanced.get('fallback'):raise ValueError('Alternate relation enhancement fell back')
            candidate=E.parse(output).getroot()
        trial=copy.deepcopy(root);changes=fuse(trial,candidate,geometry,other,image)
        if not changes:return root,geometry,report
        result=copy.deepcopy(geometry)
        for si in sorted({c['system']-1 for c in changes}):
            old=result['systems'][si]['staffs'][0];new=other['systems'][si]['staffs'][0]
            end=old['grid'][-1]['x']
            # Geometry must also retain visible heads in the formerly clipped
            # end of the last OLD bar, even though its XML is not rewritten.
            # Omitting those pixels makes later sequence alignment drift into
            # the newly appended bar. Existing heads retain their coordinates.
            if len(new['notes'])>1024:raise ValueError('Tail head geometry exceeds bounded staff size')
            for h in new['notes']:
                if h['x']<=end:continue
                if not any(abs(h['x']-v['x'])<.25*h['staff_space'] and abs(h['y']-v['y'])<.25*h['staff_space'] for v in old['notes']):old['notes'].append(copy.deepcopy(h))
            old['grid'].extend(copy.deepcopy(p) for p in new['grid'] if p['x']>end)
            old['x_range'][1]=old['grid'][-1]['x']
        report['changes']=changes;result['tail_geometry_additions']=changes
        return trial,result,report
    except Exception as exc:
        # The optional alternate path must never discard a successful primary
        # recognition when a child, parser or postprocessor fails.
        report.update(fallback=True,error=str(exc));return root,geometry,report
