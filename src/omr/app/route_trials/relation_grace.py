"""Small head-patch grace classifier; supervision is never loaded at inference.

Labels use the actual small Verovio head scale, with a corpus semantic check.
This corpus has filled grace heads and two hollow ordinary cue-sized heads.
Other grace/cue combinations require richer supervision before use.
Whole-work splits are inherited from the frozen detector manifest.
"""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from .runtime_inference import torch,probabilities,save
from .symbol_cnn import photo_truth,FACTOR
from .testset3_staff_resize_fast import resize_staves

OUT=Path('build/relation_context_20261010')


def net():
    return torch.nn.Sequential(torch.nn.Conv2d(1,8,3,padding=1),torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),torch.nn.Conv2d(8,16,3,padding=1),torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),torch.nn.Flatten(),torch.nn.Linear(16*8*8,8),
        torch.nn.ReLU(),torch.nn.Linear(8,1))


def patch(image,xy,space=16.):
    # Preserve the printed size relative to the staff, rather than resizing the
    # connected component itself (which would erase the useful size evidence).
    extent=space
    xs=np.linspace(xy[0]-extent,xy[0]+extent,32,dtype='float32')
    ys=np.linspace(xy[1]-extent,xy[1]+extent,32,dtype='float32')
    xx,yy=np.meshgrid(xs,ys)
    crop=cv2.remap(image,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    return ((255-crop.astype('float32'))/255)[None]


def dataset(normalize_spacing=False,proposals=None):
    import xml.etree.ElementTree as E
    manifest=json.loads(Path('build/route_trials_20261010/symbol_data/manifest.json').read_text(encoding='utf8'))
    checked=set()
    result={s:[] for s in ('train','validation','test')};counts={s:[0,0] for s in result}
    for row in manifest['rows']:
        if row['source'] not in checked:
            source=E.parse(row['source']).getroot()
            for n in source.findall('.//note'):
                if n.find('grace') is not None and n.findtext('type') in ('whole','half'):raise ValueError('Hollow grace heads require explicit semantic supervision')
                cue=n.find('cue') is not None or n.find('type') is not None and n.find('type').get('size')=='cue'
                if cue and n.find('grace') is None and n.findtext('type') not in ('whole','half'):raise ValueError('Filled cue heads require explicit semantic supervision')
            checked.add(row['source'])
        points=np.load(row['file'])['points'];heads=points[points[:,0]<2]
        labels=((heads[:,3]<17.7)&(heads[:,0]==0)).astype('int64')
        coords=photo_truth(row,heads)[:,1:3]
        raw=cv2.imread(row['photo'],0);image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA)
        # Cache only 32-square patches, avoiding a second copy of all full pages.
        graph=json.loads((Path('build/relation_decoder_20261010/full_final')/row['id']/(row['page']+'.graph.json')).read_text(encoding='utf8'))
        geometry=json.loads((Path('build/staff_removal_20261010/geometries')/(row['id']+'_'+row['page']+'.json')).read_text(encoding='utf8'))
        staves=resize_staves(geometry['staves'],FACTOR,FACTOR)
        predicted=[]
        for record in graph['systems']:
            selected=[s for s in staves if s['system']==record['system']][record['staff']-1]
            predicted.extend((h,selected['space_px']) for h in record['nodes'] if h['class']<2)
        if proposals is not None:
            from .relation_decode import staff_distance
            proposed=json.loads((proposals/row['id']/(row['page']+'.json')).read_text(encoding='utf8'))
            import hashlib
            if proposed['image_sha256']!=hashlib.sha256(Path(row['photo']).read_bytes()).hexdigest():raise ValueError('Grace training proposal/photo mismatch')
            for c,x,y,p in proposed['predictions']:
                if any(np.linalg.norm(np.array(h['xy'])-[x,y])<6 for h,s in predicted):continue
                eligible=[s for s in staves if s['lines'][0]['start'][0]<x<s['lines'][0]['end'][0]]
                if not eligible:continue
                staff=min(eligible,key=lambda s:staff_distance(s,x,y))
                if staff_distance(staff,x,y)>12:continue
                predicted.append(({'xy':[x,y]},staff['space_px']))
        examples=[]
        for head,space in predicted:
            distances=np.linalg.norm(coords-np.array(head['xy']),axis=1);j=int(np.argmin(distances)) if len(distances) else None
            examples.append((head['xy'],int(labels[j]) if j is not None and distances[j]<=5.6 else 0,space))
        for xy,label,space in examples:
            result[row['split']].append((patch(image,xy,space if normalize_spacing else 16.),int(label),row['id']+'_'+row['page']))
            counts[row['split']][int(label)]+=1
    return result,counts


def train(steps=500,hard_negatives=False,normalize_spacing=False,proposals=None):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(7123)
    rng=np.random.default_rng(7123);data,counts=dataset(normalize_spacing,proposals);model=net()
    x=np.stack([v[0] for v in data['train']]);y=np.array([v[1] for v in data['train']])
    vx=torch.tensor(np.stack([v[0] for v in data['validation']]));vy=torch.tensor([v[1] for v in data['validation']],dtype=torch.float32)
    by=[np.where(y==c)[0] for c in (0,1)];opt=torch.optim.AdamW(model.parameters(),lr=.001)
    OUT.mkdir(parents=True,exist_ok=True);best=float('inf');history=[];start=time.perf_counter();hard=by[0]
    for step in range(1,steps+1):
        indices=(np.r_[rng.choice(by[0],16),rng.choice(hard,16),rng.choice(by[1],32)] if hard_negatives else np.r_[rng.choice(by[0],32),rng.choice(by[1],32)]);batch=x[indices].copy()
        for i in range(len(batch)):
            m=cv2.getRotationMatrix2D((15.5,15.5),rng.uniform(-1,1),rng.uniform(.96,1.04))
            m[:,2]+=rng.uniform(-1.6,1.6,2)
            batch[i,0]=cv2.warpAffine(batch[i,0],m,(32,32))
        model.train();opt.zero_grad();logit=model(torch.tensor(batch)).flatten()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor(y[indices],dtype=torch.float32));loss.backward();opt.step()
        if step%50==0 or step==steps:
            model.eval()
            with torch.no_grad():pred=torch.cat([model(vx[i:i+128]).flatten() for i in range(0,len(vx),128)])
            # Balanced validation loss prevents the few graces being ignored.
            vl=sum(torch.nn.functional.binary_cross_entropy_with_logits(pred[vy==c],vy[vy==c]).item() for c in (0,1))/2
            history.append({'step':step,'validation_loss':vl,'elapsed_s':time.perf_counter()-start})
            if vl<best:best=vl;torch.save(model.state_dict(),OUT/'grace.pt')
            if hard_negatives and step%100==0:
                with torch.no_grad():
                    negative_logits=[]
                    for i in range(0,len(by[0]),128):
                        batch_indices=by[0][i:i+128]
                        negative_logits.append(model(torch.tensor(x[batch_indices])).flatten())
                    negative=torch.sigmoid(torch.cat(negative_logits)).numpy()
                hard=by[0][negative>=.15]
                if not len(hard):hard=by[0]
    model.load_state_dict(torch.load(OUT/'grace.pt',weights_only=True));reports={}
    probabilities={}
    for split in ('validation','test'):
        xx=torch.tensor(np.stack([v[0] for v in data[split]]));yy=np.array([v[1] for v in data[split]])
        with torch.no_grad():p=torch.sigmoid(torch.cat([model(xx[i:i+128]).flatten() for i in range(0,len(xx),128)])).numpy()
        probabilities[split]=(p,yy)
    # Select high precision before recall, using validation only.
    candidates=np.arange(.5,.991,.01)
    def rank(t):
        p,y=probabilities['validation'];tp=int(((p>=t)&(y==1)).sum());fp=int(((p>=t)&(y==0)).sum())
        return (-fp,tp,-t)
    threshold=float(max(candidates,key=rank))
    for split,(p,y) in probabilities.items():
        reports[split]={'true_positive':int(((p>=threshold)&(y==1)).sum()),'false_positive':int(((p>=threshold)&(y==0)).sum()),'reference_graces':int(y.sum()),'heads':len(y)}
    save(OUT/'grace_training.json',{'parameters':sum(p.numel() for p in model.parameters()),'patch':[1,1,32,32],'threshold':threshold,'counts':counts,'reports':reports,'history':history,'hard_negative_mining':hard_negatives,'normalize_patch_to_staff_spacing':normalize_spacing,'low_threshold_proposals':str(proposals) if proposals else None,'test_labels_consumed_for_training_or_threshold':False,'scope':'Frozen image-predicted head centres matched to supervision within 5.6 pixels. Source semantics checked: all grace heads filled; the two hollow cue-sized ordinary heads labelled negative. Filled cues/hollow graces unsupported.'})
    print(json.dumps({'threshold':threshold,'reports':reports,'counts':counts}),flush=True)


def load(root=OUT):
    model=net();model.load_state_dict(torch.load(root/'grace.pt',map_location='cpu',weights_only=True));model.eval()
    return model,json.loads((root/'grace_training.json').read_text(encoding='utf8'))['threshold']


def classify_groups(groups,image,staff,model,threshold):
    heads=[h for g in groups for h in g.get('heads',[])];space=staff['space_px']
    if not heads:return
    x=np.stack([patch(image,h['xy'],space) for h in heads])
    p=np.concatenate([probabilities(model,x[i:i+128]).flatten() for i in range(0,len(x),128)])
    for h,v in zip(heads,p):
        h['grace_probability']=float(v)
        h['head_width_spaces']=head_width(image,h['xy'],space)
    for g in groups:
        hs=g.get('heads',[])
        g['grace']=bool(hs and all(h['grace_probability']>=threshold and .65<h['head_width_spaces']<1.12 for h in hs))


def head_width(image,xy,space=16.,quantile=1.):
    """Maximum supported head row, excluding staff/ledger lines and stems.

    A model vote alone can confuse tightly overlapping chord tones with small
    notes. Require independent printed-width evidence before dropping time.
    """
    x,y=xy;rows=[]
    x0=max(0,round(x-1.1*space));x1=min(image.shape[1],round(x+1.1*space)+1)
    centre=round(x)-x0
    for yy in range(max(0,round(y-.32*space)),min(image.shape[0],round(y+.32*space)+1)):
        a=image[yy,x0:x1]<155
        if not len(a):continue
        edges=np.diff(np.r_[False,a,False].astype('int8'));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
        spans=[(b-a) for a,b in zip(starts,ends) if a-.15*space<=centre<b+.15*space and .55*space<b-a<1.65*space]
        if spans:rows.append(max(spans)/space)
    return float(np.quantile(rows,quantile)) if rows else 0.


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=500);parser.add_argument('--hard-negatives',action='store_true');parser.add_argument('--normalize-spacing',action='store_true');parser.add_argument('--proposals',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output:OUT=args.output
    train(args.steps,args.hard_negatives,args.normalize_spacing,args.proposals)
