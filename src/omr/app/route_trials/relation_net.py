"""Bounded stroke segmentation plus a learned head--stem edge classifier."""
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from .runtime_inference import torch,probabilities
from experimental_omr.cnn import build_net
from .symbol_cnn import H, W, FACTOR

CLASSES=['stem','beam','barline','flag_1','flag_2','flag_3']


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    compact=path.name.endswith(('.graph.json','.staff.json'))
    options={'indent':None,'separators':(',',':')} if compact else {'indent':2}
    path.write_text(json.dumps(data,**options)+'\n',encoding='utf8')


def load(manifest, split):
    rows=[]
    for row in manifest['rows']:
        if row['split']!=split:continue
        z=np.load(row['file']);packed=z['packed'];centres=[]
        for c in range(6):
            yy,xx=np.where(packed & (1<<c));indices=np.linspace(0,len(xx)-1,min(1000,len(xx)),dtype=int)
            centres.append(np.c_[xx[indices],yy[indices]])
        rows.append((z['image'],packed,centres))
    return rows


def patch(data, rng, augment=True):
    active=[c for c in range(6) if any(len(d[2][c]) for d in data)]
    c=int(rng.choice(active));eligible=[d for d in data if len(d[2][c])]
    image,packed,centres=eligible[int(rng.integers(len(eligible)))];p=centres[c][int(rng.integers(len(centres[c])))]
    x0=int(p[0]-rng.uniform(32,W-32));y0=int(p[1]-rng.uniform(24,H-24))
    crop=np.full((H,W),255,np.uint8);bits=np.zeros((H,W),np.uint8)
    x1,y1=max(0,x0),max(0,y0);x2,y2=min(image.shape[1],x0+W),min(image.shape[0],y0+H)
    crop[y1-y0:y2-y0,x1-x0:x2-x0]=image[y1:y2,x1:x2]
    bits[y1-y0:y2-y0,x1-x0:x2-x0]=packed[y1:y2,x1:x2]
    target=np.stack([(bits & (1<<c))!=0 for c in range(6)]).astype('uint8')
    if augment:
        m=cv2.getRotationMatrix2D((W/2,H/2),rng.uniform(-.65,.65),rng.uniform(.92,1.08))
        crop=cv2.warpAffine(crop,m,(W,H),borderValue=255)
        target=np.stack([cv2.warpAffine(t,m,(W,H),flags=cv2.INTER_NEAREST) for t in target])
        if rng.random()<.3:crop=cv2.GaussianBlur(crop,(3,3),rng.uniform(.1,.6))
    # Output centres at input 2*j; no resize half-pixel shift.
    target=target[:,::2,::2].astype('float32')
    ink=(255-crop.astype('float32'))/255
    if augment:ink=np.clip(ink*rng.uniform(.8,1.05)+rng.normal(0,.004,ink.shape),0,1).astype('float32')
    return ink[None],target


def loss(logit, target):
    axes=(0,2,3);p=torch.sigmoid(logit)
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,target,reduction='none')
    weights=1+7*target
    b=(bce*weights).sum(axes)/weights.sum(axes)
    dice=1-(2*(p*target).sum(axes)+1)/(p.sum(axes)+target.sum(axes)+1)
    return (b+.7*dice).mean()


def train(manifest, out, steps, batch):
    out.mkdir(parents=True,exist_ok=True);torch.manual_seed(7123);rng=np.random.default_rng(7123)
    data=load(manifest,'train');validation=load(manifest,'validation')
    net=build_net(n_out=6,ch=12);net.head.bias.data.fill_(-2.2)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
    fixed=[patch(validation,np.random.default_rng(9200+i),False) for i in range(32)]
    vx=torch.tensor(np.stack([p[0] for p in fixed]));vy=torch.tensor(np.stack([p[1] for p in fixed]))
    best=float('inf');history=[];start=time.perf_counter();running=0
    for step in range(1,steps+1):
        net.train();pairs=[patch(data,rng) for _ in range(batch)]
        x=torch.tensor(np.stack([p[0] for p in pairs]));y=torch.tensor(np.stack([p[1] for p in pairs]))
        optimizer.zero_grad();v=loss(net(x),y);v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);optimizer.step();running+=v.item()
        if step%100==0 or step==steps:
            net.eval()
            with torch.no_grad():vl=sum(loss(net(vx[i:i+8]),vy[i:i+8]).item() for i in range(0,32,8))/4
            record={'step':step,'train_loss':running/(100 if step%100==0 else step%100),'validation_loss':vl,'elapsed_s':time.perf_counter()-start}
            history.append(record);running=0;print(json.dumps(record),flush=True)
            state={'model':net.state_dict(),'channels':12,'step':step,'classes':CLASSES}
            torch.save(state,out/'last.pt')
            if vl<best:best=vl;torch.save(state,out/'best.pt')
            save(out/'training.json',{'parameters':sum(p.numel() for p in net.parameters()),'seed':7123,'history':history,'train_pages':len(data),'validation_pages':len(validation),'test_pages_consumed':0,'device':'cpu'})


def open_net(path):
    state=torch.load(path,map_location='cpu',weights_only=True);net=build_net(n_out=6,ch=state['channels']);net.load_state_dict(state['model']);net.eval()
    return net,state


def predict(net, image):
    h,w=image.shape;result=np.zeros((6,h,w),np.uint8)
    for y in range(0,h,H-32):
        for x in range(0,w,W-32):
            tile=np.full((H,W),255,np.uint8);a=image[y:y+H,x:x+W];tile[:a.shape[0],:a.shape[1]]=a
            p=probabilities(net,((255-tile.astype('float32'))/255)[None,None])[0]
            # Sample output lattice at input coordinates, retaining overlap interior.
            left=16 if x else 0;top=16 if y else 0;right=min(W-16 if x+W<w else W,w-x);bottom=min(H-16 if y+H<h else H,h-y)
            for c in range(6):
                full=cv2.resize(p[c],(W,H),interpolation=cv2.INTER_NEAREST)
                result[c,y+top:y+bottom,x+left:x+right]=np.rint(full[top:bottom,left:right]*255).astype('uint8')
    return result


def edge_features(head, stem, space=16.):
    hx,hy=head; a,b=np.array(stem);x=(a[0]+b[0])/2;lo,hi=sorted((a[1],b[1]))
    dx=(x-hx)/space;above=(hy-lo)/space;below=(hi-hy)/space
    return [dx,abs(dx),above,below,(hi-lo)/space,max(lo-hy,hy-hi,0)/space,
            min(abs(hy-lo),abs(hy-hi))/space,abs(abs(dx)-.55)]


def edge_net():
    return torch.nn.Sequential(torch.nn.Linear(8,24),torch.nn.ReLU(),torch.nn.Linear(24,12),torch.nn.ReLU(),torch.nn.Linear(12,1))


def edge_data(manifest,split,jitter=False):
    rng=np.random.default_rng(913);features=[];labels=[]
    for row in manifest['rows']:
        if row['split']!=split:continue
        graph=json.loads(Path(row['graph']).read_text(encoding='utf8'));positive={tuple(p) for p in graph['head_stem_edges']}
        for head in graph['heads']:
            for stem in graph['stems']:
                hx,hy=head['xy'];a,b=np.array(stem['xy']);sx=(a[0]+b[0])/2;lo,hi=sorted((a[1],b[1]))
                if abs(sx-hx)>1.8*16 or not lo-8<=hy<=hi+8:continue
                for _ in range(3 if jitter else 1):
                    hp=np.array(head['xy']);sp=np.array(stem['xy'])
                    if jitter:hp+=rng.normal(0,1.1,2);sp+=rng.normal(0,1.1,(2,2))
                    features.append(edge_features(hp,sp));labels.append(float((head['id'],stem['id']) in positive))
    return torch.tensor(np.array(features,dtype='float32')),torch.tensor(labels,dtype=torch.float32)[:,None]


def train_edges(manifest,out):
    torch.manual_seed(7123);net=edge_net();x,y=edge_data(manifest,'train',True);vx,vy=edge_data(manifest,'validation')
    opt=torch.optim.AdamW(net.parameters(),lr=.003);best=float('inf');history=[]
    for step in range(400):
        ix=torch.randint(len(x),(512,));pred=net(x[ix]);v=torch.nn.functional.binary_cross_entropy_with_logits(pred,y[ix]);opt.zero_grad();v.backward();opt.step()
        if (step+1)%25==0:
            with torch.no_grad():vl=torch.nn.functional.binary_cross_entropy_with_logits(net(vx),vy).item()
            history.append({'step':step+1,'validation_loss':vl})
            if vl<best:best=vl;torch.save(net.state_dict(),out/'edge.pt')
    net.load_state_dict(torch.load(out/'edge.pt',weights_only=True))
    with torch.no_grad():vp=torch.sigmoid(net(vx)).numpy().ravel()
    labels=vy.numpy().ravel()>0;thresholds=np.arange(.1,.96,.05)
    threshold=max(thresholds,key=lambda t:2*((vp>=t)&labels).sum()/max(1,(vp>=t).sum()+labels.sum()))
    reports={}
    for split in ('validation','test'):
        tx,ty=edge_data(manifest,split)
        with torch.no_grad():p=torch.sigmoid(net(tx)).numpy().ravel()>=threshold
        t=ty.numpy().ravel()>0;tp=int((p&t).sum())
        full_reference=sum(len(json.loads(Path(r['graph']).read_text(encoding='utf8'))['head_stem_edges']) for r in manifest['rows'] if r['split']==split)
        reports[split]={'matched':tp,'reference':full_reference,'eligible_reference':int(t.sum()),'candidate':int(p.sum()),'recall':tp/max(1,full_reference),'eligible_recall':tp/max(1,int(t.sum())),'precision':tp/max(1,int(p.sum()))}
    save(out/'edge_training.json',{'parameters':sum(p.numel() for p in net.parameters()),'train_pairs_with_jitter':len(x),'validation_pairs':len(vx),'history':history,'threshold':float(threshold),'threshold_selected_on':'validation only','reports':reports,'scope':'True SVG nodes with jitter during training; this is not predicted-node end-to-end accuracy.'})


def score(manifest,model,out):
    out.mkdir(parents=True,exist_ok=True);net,state=open_net(model)
    thresholds=[.2,.3,.4,.5,.6,.7];totals={s:{t:np.zeros((6,3),np.int64) for t in thresholds} for s in ('validation','test')}
    for row in manifest['rows']:
        if row['split'] not in totals:continue
        z=np.load(row['file']);p=predict(net,z['image']);bits=z['packed'];truth=np.stack([(bits&(1<<c))!=0 for c in range(6)])
        for threshold in thresholds:
            pred=p>=round(threshold*255)
            totals[row['split']][threshold]+=np.stack([(pred&truth).sum((1,2)),truth.sum((1,2)),pred.sum((1,2))],axis=1)
        np.savez_compressed(out/(row['id']+'_'+row['page']+'.npz'),probabilities=p)
        print(row['id'],row['page'],flush=True)
    chosen=[max(thresholds,key=lambda t:2*totals['validation'][t][c,0]/max(1,sum(totals['validation'][t][c,1:]))) for c in range(6)]
    reports={}
    for split in totals:
        counts=np.array([totals[split][chosen[c]][c] for c in range(6)])
        reports[split]={name:{'intersection':int(counts[c,0]),'reference_pixels':int(counts[c,1]),'candidate_pixels':int(counts[c,2]),'pixel_f1':float(2*counts[c,0]/max(1,sum(counts[c,1:]))),'threshold':chosen[c]} for c,name in enumerate(CLASSES)}
    save(out/'scores.json',{'checkpoint_step':state['step'],'reports':reports,'threshold_selected_on':'validation only','scope':'Stroke pixels only, not music-event accuracy; stem/bar labels dilated one pixel at normalized scale.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','edges','score']);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--model',type=Path);p.add_argument('--steps',type=int,default=1600);p.add_argument('--batch',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);manifest=json.loads(a.manifest.read_text(encoding='utf8'))
    if a.mode=='train':train(manifest,a.output,a.steps,a.batch)
    elif a.mode=='edges':train_edges(manifest,a.output)
    else:score(manifest,a.model,a.output)


if __name__=='__main__':main()
