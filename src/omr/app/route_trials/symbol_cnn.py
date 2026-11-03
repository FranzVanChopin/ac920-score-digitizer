"""Small-CNN training experiments and shared Torch/ONNX CPU inference.

Train from clean SVG-labelled pages with random image degradation; select
checkpoint and thresholds only on validation compositions. Test photo labels
are transformed with the recorded camera matrix only inside the scorer.
"""
import argparse,json,time,random,hashlib
from pathlib import Path
import cv2
import numpy as np
from .runtime_inference import torch,probabilities
from experimental_omr.cnn import build_net

FACTOR=16/19.8
H,W=160,320

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def load_data(manifest,split):
    data=[]
    for row in manifest['rows']:
        if row['split']==split:
            z=np.load(row['file']);data.append((z['image'],z['points'],row))
    return data

def patch(data,rng,augment=True):
    # Balance classes by choosing a page/symbol rather than image area.
    active=[c for c in range(12) if any(np.any(p[:,0]==c) for _,p,_ in data)]
    c=int(rng.choice(active));eligible=[d for d in data if np.any(d[1][:,0]==c)]
    image,points,_=eligible[int(rng.integers(len(eligible)))];point=points[points[:,0]==c][int(rng.integers(sum(points[:,0]==c)))]
    x0=int(point[1]-rng.uniform(28,W-28));y0=int(point[2]-rng.uniform(28,H-28))
    crop=np.full((H,W),255,np.uint8);x1,y1=max(0,x0),max(0,y0);x2,y2=min(image.shape[1],x0+W),min(image.shape[0],y0+H)
    if x2>x1 and y2>y1:crop[y1-y0:y2-y0,x1-x0:x2-x0]=image[y1:y2,x1:x2]
    pts=points.copy();pts[:,1]-=x0;pts[:,2]-=y0
    if augment:
        m=cv2.getRotationMatrix2D((W/2,H/2),rng.uniform(-.65,.65),rng.uniform(.92,1.08))
        crop=cv2.warpAffine(crop,m,(W,H),borderValue=255)
        pts[:,1:3]=pts[:,1:3]@m[:,:2].T+m[:,2]
        if rng.random()<.3:crop=cv2.GaussianBlur(crop,(3,3),rng.uniform(.1,.6))
    ink=(255-crop.astype('float32'))/255
    if augment:ink=np.clip(ink*rng.uniform(.75,1.05)+rng.normal(0,.005,ink.shape),0,1).astype('float32')
    target=np.zeros((12,H//2,W//2),np.float32)
    for c,x,y,_,_ in pts:
        # Stride-two convolution output centres are at 2*j (padding=1).
        if 4<x<W-4 and 4<y<H-4:
            cx,cy=int(round(x/2)),int(round(y/2));c=int(c)
            lo,hi=max(0,cx-4),min(W//2,cx+5);top,bottom=max(0,cy-4),min(H//2,cy+5)
            yy,xx=np.mgrid[top:bottom,lo:hi]
            g=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*1.1**2))
            target[c,top:bottom,lo:hi]=np.maximum(target[c,top:bottom,lo:hi],g)
    return ink[None],target

def loss(logit,target):
    p=torch.sigmoid(logit).clamp(1e-5,1-1e-5);positive=target.eq(1)
    pos=-torch.log(p)*(1-p)**2*positive
    neg=-torch.log(1-p)*p**2*(1-target)**4*(~positive)
    # Normalize each class independently: dense filled heads must not suppress
    # the few half/short rests sharing the same crop.
    axes=(0,2,3)
    return ((pos.sum(axes)+.25*neg.sum(axes))/positive.sum(axes).clamp_min(1)).mean()

def train(manifest,out,steps,batch):
    out.mkdir(parents=True,exist_ok=True);torch.manual_seed(7123);rng=np.random.default_rng(7123)
    training=load_data(manifest,'train');validation=load_data(manifest,'validation')
    net=build_net(n_out=12,ch=12);net.head.bias.data.fill_(-2.2)
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
    fixed=[patch(validation,np.random.default_rng(9000+i),False) for i in range(48)]
    vx=torch.tensor(np.stack([x for x,_ in fixed]));vy=torch.tensor(np.stack([y for _,y in fixed]))
    history=[];best=float('inf');start=time.perf_counter();running=0.
    for step in range(1,steps+1):
        net.train();pairs=[patch(training,rng) for _ in range(batch)]
        x=torch.tensor(np.stack([x for x,_ in pairs]));y=torch.tensor(np.stack([y for _,y in pairs]))
        opt.zero_grad();v=loss(net(x),y);v.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step();running+=v.item()
        if step%100==0 or step==steps:
            net.eval()
            with torch.no_grad():vl=sum(loss(net(vx[i:i+8]),vy[i:i+8]).item() for i in range(0,48,8))/6
            row={'step':step,'train_loss':running/(100 if step%100==0 else step%100),'validation_loss':vl,'elapsed_s':time.perf_counter()-start}
            history.append(row);print(json.dumps(row),flush=True);running=0.
            state={'model':net.state_dict(),'channels':12,'classes':manifest['classes'],'step':step}
            torch.save(state,out/'last.pt')
            if vl<best:best=vl;torch.save(state,out/'best.pt')
            save(out/'training.json',{'parameters':sum(p.numel() for p in net.parameters()),'seed':7123,'history':history,'train_pages':len(training),'validation_pages':len(validation),'test_pages_consumed':0,'device':'cpu','threads':torch.get_num_threads()})
    torch.save(opt.state_dict(),out/'optimizer.pt')

def open_net(path):
    state=torch.load(path,map_location='cpu',weights_only=True);net=build_net(n_out=12,ch=state['channels']);net.load_state_dict(state['model']);net.eval();return net,state

def predict(net,image,threshold=.15):
    h,w=image.shape;result=[]
    # Fixed-size streaming tiles. Discard overlap edges, then global NMS.
    for y in range(0,h, H-32):
        for x in range(0,w,W-32):
            tile=np.full((H,W),255,np.uint8);a=image[y:y+H,x:x+W];tile[:a.shape[0],:a.shape[1]]=a
            p=probabilities(net,((255-tile.astype('float32'))/255)[None,None])[0]
            for c in range(12):
                peak=(p[c]>=threshold)&(p[c]==cv2.dilate(p[c],np.ones((5,5),np.uint8)))
                yy,xx=np.where(peak)
                for cy,cx in zip(yy,xx):
                    px,py=2*cx,2*cy
                    if (x and px<16) or (y and py<16) or (x+W<w and px>=W-16) or (y+H<h and py>=H-16):continue
                    if x+px<w and y+py<h:result.append([c,x+px,y+py,float(p[c,cy,cx])])
    # Duplicate-free one-to-one detections, no label access.
    final=[]
    for p in sorted(result,key=lambda p:-p[3]):
        if not any(p[0]==q[0] and (p[1]-q[1])**2+(p[2]-q[2])**2<36 for q in final):final.append(p)
    return np.array(final,dtype='float32').reshape(-1,4)

def photo_truth(row,points):
    camera=json.loads((Path(row['photo']).parent/'page.json').read_text(encoding='utf8'))['camera']
    p=points.copy();xy=p[:,1:3]/FACTOR+camera['clean_paste_xy_px']
    hom=np.array(camera['homography_before_final_rotation']);v=np.c_[xy,np.ones(len(xy))]@hom.T;xy=v[:,:2]/v[:,2:3]
    w,h=camera['photo_size_px'];angle=camera['degradation']['rotation_deg']
    m=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
    p[:,1:3]=(xy@m[:,:2].T+m[:,2])*FACTOR
    return p

def match(truth,pred,threshold):
    counts=np.zeros((12,3),int)
    for c in range(12):
        t=truth[truth[:,0]==c,1:3];p=pred[(pred[:,0]==c)&(pred[:,3]>=threshold),1:3]
        counts[c,1:]=[len(t),len(p)]
        pairs=[]
        for i,xy in enumerate(t):
            distances=np.linalg.norm(p-xy,axis=1)
            pairs.extend((float(distances[j]),i,j) for j in np.where(distances<=5.6)[0])
        ti=set();pi=set()
        for _,i,j in sorted(pairs):
            if i not in ti and j not in pi:ti.add(i);pi.add(j)
        counts[c,0]=len(ti)
    return counts

def score(manifest,checkpoint,out):
    out.mkdir(parents=True,exist_ok=True);net,state=open_net(checkpoint);start=time.perf_counter()
    all_predictions={};photo_labels={};thresholds=[.15,.2,.25,.3,.35,.4,.45,.5,.55,.6]
    totals={split:{t:np.zeros((12,3),int) for t in thresholds} for split in ('validation','test')}
    for row in manifest['rows']:
        if row['split'] not in totals:continue
        raw=cv2.imread(row['photo'],0);image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA)
        key=row['id']+'_'+row['page'];pred=predict(net,image);pts=photo_truth(row,np.load(row['file'])['points'])
        all_predictions[key]=pred.tolist();photo_labels[key]=pts.tolist()
        for t in thresholds:totals[row['split']][t]+=match(pts,pred,t)
        print(json.dumps({'id':key,'symbols':len(pts),'proposals':len(pred),'elapsed_s':time.perf_counter()-start}),flush=True)
    # Per-class threshold selection uses validation only, no test scores.
    chosen=[]
    for c in range(12):
        chosen.append(max(thresholds,key=lambda t:2*totals['validation'][t][c,0]/max(1,sum(totals['validation'][t][c,1:]))))
    reports={}
    for split in totals:
        counts=np.array([totals[split][chosen[c]][c] for c in range(12)])
        reports[split]={'matched':int(counts[:,0].sum()),'reference':int(counts[:,1].sum()),'candidate':int(counts[:,2].sum()),
                        'classes':{name:{'matched':int(counts[c,0]),'reference':int(counts[c,1]),'candidate':int(counts[c,2]),'threshold':chosen[c]} for c,name in enumerate(manifest['classes'])}}
        n=reports[split];n['recall']=n['matched']/max(1,n['reference']);n['precision']=n['matched']/max(1,n['candidate'])
    save(out/'scores.json',{'checkpoint_step':state['step'],'threshold_selected_on':'validation compositions only','tolerance_pixels':5.6,'normalized_staff_space':16,'elapsed_s':time.perf_counter()-start,'reports':reports,
        'limitations':['Symbol localization/classification metrics are not full note-event accuracy.','Single engraving font; simulated clear-lit photographs; all scores used for previous project development.','No tie, slur, beam, voice, key-change or clef classes trained.','Camera ground truth is used in scorer only, never supplied to detector.']})
    save(out/'predictions.json',all_predictions)
    # One label overlay for audit, reference vs predicted centres.
    row=next(r for r in manifest['rows'] if r['id']=='score35' and r['page']=='page001')
    key=row['id']+'_'+row['page'];raw=cv2.imread(row['photo']);raw=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR)
    for c,x,y,*_ in photo_labels[key]:cv2.circle(raw,(round(x),round(y)),4,(0,180,0),1)
    for c,x,y,s in all_predictions[key]:
        if s>=chosen[int(c)]:cv2.circle(raw,(round(x),round(y)),6,(0,0,230),1)
    cv2.imwrite(str(out/'overlay.png'),raw)
    print(json.dumps(reports,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','score']);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch',type=int,default=8);p.add_argument('--checkpoint',type=Path);args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manifest=json.loads(args.manifest.read_text(encoding='utf8'))
    if args.mode=='train':train(manifest,args.output,args.steps,args.batch)
    else:score(manifest,args.checkpoint,args.output)

if __name__=='__main__':main()
