#!/usr/bin/env python3
"""Residual-Attention U-Net: complete executable reference implementation.

INSTALL (Python 3.10+):
    python -m pip install torch numpy pillow opencv-python-headless

RUN A SELF-CONTAINED CPU DEMO:
    python ra_unet_hiseg623.py

TRAIN ON ANNOTATED IMAGES:
    python ra_unet_hiseg623.py --mode train --manifest dataset.csv --output real_run

FIVE-FOLD CROSS-VALIDATION, REPEATED FIVE TIMES:
    python ra_unet_hiseg623.py --mode cv --manifest dataset.csv --repeats 5 --output cv_run

PREDICT:
    python ra_unet_hiseg623.py --mode predict --checkpoint real_run/best.pt --image example.png --output prediction

MANIFEST (UTF-8 CSV): image,mask,group_id
Paths are relative to the manifest directory or absolute. Each crop from the same
source image must share group_id. Masks must be single-channel class-index PNGs:
0=Person, 1=Architecture, 2=Text, 3=Flora & Fauna, 4=Ornament. Unknown/background
pixels must be 255 (ignored), NOT class 0. RGB visualization masks are rejected.
The previously supplied image-only HISeg-623 archive is insufficient for training.

METHOD AND REPRODUCIBILITY:
Based on Section 3.2 of 'Image Semantic Segmentation and Inquiry Learning in the
Teaching of Historical Humanities with U-Net Empowerment'. Paper defaults:
512x512 input; channels 64/128/256/512/1024; four down/up stages; residual double
3x3 convolutions; spatial attention gates; 2x2 transposed convolutions; BN/ReLU;
five classes; 0.5*Dice+0.5*CE; AdamW(lr=1e-4, weight_decay=1e-5), batch 8,
betas=(0.9,0.999); patience 15; seed 42; specified image augmentations and CLAHE.

Explicit implementation choices where the paper is incomplete:
* '200 iterations' is interpreted as 200 epochs (override --epochs).
* Residual decoder blocks follow paragraph 34; projection shortcuts are used
  when channel counts change. Counting projections, gates and output layers
  gives more than the stated 24 convolutions; the count is printed, not hidden.
* The ImageNet checkpoint/backbone mapping is not supplied. Initialization is
  random unless --encoder-weights supplies an EXACT compatible encoder state
  dictionary. No unrelated ImageNet weights are silently mapped or claimed.
* CLAHE uses LAB luminance, clipLimit=2, tileGridSize=(8,8); normalization is
  ImageNet mean/std. Dice is a per-image, per-class soft Dice average.
* Unlabelled pixels use ignore_index=255. Train/validation/test splitting is
  source-group disjoint. Validation selects checkpoints; test is held out.
* CV repeats use seeds 42+r and fresh group partitions. The validation subset
  is drawn inside each training fold; fold test data never selects checkpoints.
* Demo uses 24 generated geometric examples, width 8, 64x64 inputs and 2 epochs.
  It tests execution only. It is not historical data or a replication of 89.7%.

OUTPUTS:
config.json, splits.json, model.txt, training.csv, best.pt, metrics.json,
confusion_matrix.csv, per-image class masks/color maps/overlays/probabilities,
and CV fold/repeat summaries when requested. All reported metrics are computed
from predictions; no published scores are inserted as measured results.
Existing output directories are refused to protect previous runs.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

CLASSES = ['Person', 'Architecture', 'Text', 'Flora & Fauna', 'Ornament']
COLORS = np.array([[220,60,60],[60,110,225],[235,190,40],[50,165,95],[160,80,195]],dtype=np.uint8)
IGNORE = 255
MEAN = torch.tensor([.485,.456,.406])[:,None,None]
STD = torch.tensor([.229,.224,.225])[:,None,None]

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
                                nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout))
        self.shortcut=nn.Identity() if cin==cout else nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x): return F.relu(self.body(x)+self.shortcut(x),inplace=True)

class AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        middle=max(1,channels//2)
        self.skip=nn.Sequential(nn.Conv2d(channels,middle,1,bias=False),nn.BatchNorm2d(middle))
        self.gate=nn.Sequential(nn.Conv2d(channels,middle,1,bias=False),nn.BatchNorm2d(middle))
        self.score=nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(middle,1,1),nn.Sigmoid())
    def forward(self,x,g): return x*self.score(self.skip(x)+self.gate(g))

class UpBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.up=nn.ConvTranspose2d(cin,cout,2,stride=2)
        self.attention=AttentionGate(cout)
        self.fuse=ResidualBlock(cout*2,cout)
    def forward(self,x,skip):
        x=self.up(x)
        if x.shape[-2:]!=skip.shape[-2:]:x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.fuse(torch.cat([x,self.attention(skip,x)],dim=1))

class RAUNet(nn.Module):
    def __init__(self,base=64):
        super().__init__()
        widths=[base*2**i for i in range(5)]
        self.encoder=nn.ModuleList([ResidualBlock(3,widths[0])]+[ResidualBlock(widths[i-1],widths[i]) for i in range(1,5)])
        self.decoder=nn.ModuleList([UpBlock(widths[i],widths[i-1]) for i in range(4,0,-1)])
        self.head=nn.Conv2d(base,5,1)
    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.encoder):
            if i:x=F.max_pool2d(x,2)
            x=block(x);skips.append(x)
        for block,skip in zip(self.decoder,reversed(skips[:-1])):x=block(x,skip)
        return self.head(x)

def load_tensors(path,device='cpu'):
    # weights_only avoids executing arbitrary Python objects from checkpoints.
    return torch.load(path,map_location=device,weights_only=True)

def preprocess(image,size):
    rgb=np.array(image.convert('RGB').resize((size,size),Image.Resampling.BILINEAR))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
    return Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))

def image_tensor(image):
    return (torch.from_numpy(np.array(image,dtype=np.float32)/255).permute(2,0,1)-MEAN)/STD

class SegmentationData(Dataset):
    def __init__(self,records,size,augment=False):self.records,self.size,self.augment=records,size,augment
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        with Image.open(row['image']) as f:image=preprocess(f,self.size)
        with Image.open(row['mask']) as f:
            arr=np.array(f)
            if arr.ndim!=2:raise ValueError('Masks must be single-channel class indices: '+row['mask'])
            if not np.isin(arr,[0,1,2,3,4,IGNORE]).all():raise ValueError('Mask values must be 0..4 or 255: '+row['mask'])
            mask=Image.fromarray(arr.astype(np.uint8)).resize((self.size,self.size),Image.Resampling.NEAREST)
        if self.augment:
            if random.random()<.5:
                image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT);mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle=random.uniform(-15,15)
            image=image.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
            mask=mask.rotate(angle,resample=Image.Resampling.NEAREST,fillcolor=IGNORE)
            pixels=np.array(image,dtype=np.float32)/255
            pixels=np.clip(pixels*random.uniform(.8,1.2)+np.random.normal(0,.01,pixels.shape),0,1)
            image=Image.fromarray(np.rint(pixels*255).astype(np.uint8))
        return image_tensor(image),torch.from_numpy(np.array(mask,dtype=np.int64)),index

def hybrid_loss(logits,target):
    valid=target!=IGNORE
    if not valid.any():return logits.sum()*0
    ce=F.cross_entropy(logits,target,ignore_index=IGNORE)
    safe=target.masked_fill(~valid,0)
    truth=F.one_hot(safe,num_classes=5).permute(0,3,1,2).float()*valid[:,None]
    probs=logits.softmax(1)*valid[:,None]
    dice=(2*(probs*truth).sum((2,3))+1e-6)/(probs.sum((2,3))+truth.sum((2,3))+1e-6)
    present=valid.flatten(1).any(1)
    return .5*ce+.5*(1-dice[present].mean())

def metric_report(cm):
    tp=np.diag(cm).astype(float);actual=cm.sum(1);predicted=cm.sum(0)
    def ratio(a,b):return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b!=0)
    iou=ratio(tp,actual+predicted-tp);precision=ratio(tp,predicted);recall=ratio(tp,actual);f1=ratio(2*tp,actual+predicted)
    def clean(x):return None if not np.isfinite(x) else float(x)
    def avg(a):return clean(np.nanmean(a)) if np.isfinite(a).any() else None
    return {'mIoU':avg(iou),'macro_precision':avg(precision),'macro_recall':avg(recall),'macro_F1':avg(f1),
            'pixel_accuracy':clean(tp.sum()/cm.sum()) if cm.sum() else None,
            'averaging':'Global pixel confusion matrix; undefined class metrics excluded from their macro averages; fractions, not percentages.',
            'per_class':[{ 'class':name,'IoU':clean(iou[i]),'precision':clean(precision[i]),'recall':clean(recall[i]),'F1':clean(f1[i]),'support_pixels':int(actual[i])} for i,name in enumerate(CLASSES)]}

def save_prediction(logits,original,directory,stem):
    directory.mkdir(parents=True,exist_ok=True)
    logits=F.interpolate(logits[None],size=(original.height,original.width),mode='bilinear',align_corners=False)[0]
    probability=logits.softmax(0).cpu().numpy()
    mask=probability.argmax(0).astype(np.uint8)
    color=Image.fromarray(COLORS[mask])
    Image.fromarray(mask).save(directory/(stem+'_mask.png'))
    color.save(directory/(stem+'_color.png'))
    Image.blend(original.convert('RGB'),color,.45).save(directory/(stem+'_overlay.png'))
    np.savez_compressed(directory/(stem+'_probabilities.npz'),probabilities=probability.astype(np.float32))

@torch.no_grad()
def evaluate(model,loader,device,prediction_dir=None):
    model.eval();cm=np.zeros((5,5),dtype=np.int64);loss_total=0.;n=0
    for images,masks,indices in loader:
        images,masks=images.to(device),masks.to(device)
        logits=model(images);loss_total+=float(hybrid_loss(logits,masks))*len(images);n+=len(images)
        prediction=logits.argmax(1);valid=masks!=IGNORE
        cm+=torch.bincount((masks[valid]*5+prediction[valid]).cpu(),minlength=25).reshape(5,5).numpy()
        if prediction_dir:
            for j,index in enumerate(indices.tolist()):
                row=loader.dataset.records[index]
                with Image.open(row['image']) as original:
                    save_prediction(logits[j],original,prediction_dir,f'{index:05d}_{Path(row["image"]).stem}')
    if not cm.sum():raise ValueError('Evaluation contains no labelled pixels.')
    report=metric_report(cm);report['loss']=loss_total/n
    return report,cm

def read_manifest(path):
    path=Path(path).resolve()
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'image','mask','group_id'}.issubset(reader.fieldnames or []):raise ValueError('Manifest requires image,mask,group_id')
        rows=list(reader)
    seen=set()
    for r in rows:
        if not r['group_id'].strip():raise ValueError('Every image requires a source group_id')
        for key in ['image','mask']:
            p=(path.parent/r[key]).resolve()
            if not p.is_file():raise FileNotFoundError(p)
            r[key]=str(p)
        if r['image'] in seen:raise ValueError('Duplicate image in manifest: '+r['image'])
        seen.add(r['image'])
        with Image.open(r['image']) as im,Image.open(r['mask']) as mask:
            if im.size!=mask.size:raise ValueError('Image/mask dimensions differ: '+r['image'])
    if not rows:raise ValueError('Empty manifest')
    return rows

def partition(rows,seed,fold=None):
    groups=sorted({r['group_id'] for r in rows})
    if len(groups)<7:raise ValueError('At least seven distinct source groups are required.')
    random.Random(seed).shuffle(groups)
    if fold is None:test=set(groups[:max(1,round(.15*len(groups)))])
    else:test=set(groups[fold::5])
    remaining=[g for g in groups if g not in test]
    val=set(remaining[:max(1,round(.2*len(remaining)))])
    train=set(remaining)-val
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    return [[r for r in rows if r['group_id'] in ids] for ids in [train,val,test]]

def run_training(args,split,out,seed):
    out.mkdir(parents=True,exist_ok=False);seed_all(seed)
    save_json(out/'config.json',{**vars(args),'output':str(out),'run_seed':seed,'classes':CLASSES,
        'versions':{'torch':str(torch.__version__),'numpy':np.__version__,'opencv':cv2.__version__},
        'initialization':'compatible supplied encoder' if args.encoder_weights else 'random; manuscript ImageNet weights unavailable'})
    save_json(out/'splits.json',dict(zip(['train','validation','test'],split)))
    loaders=[DataLoader(SegmentationData(rows,args.size,i==0),batch_size=args.batch_size,shuffle=i==0,num_workers=0) for i,rows in enumerate(split)]
    model=RAUNet(args.base)
    if args.encoder_weights:model.encoder.load_state_dict(load_tensors(args.encoder_weights),strict=True)
    model=model.to(args.device)
    (out/'model.txt').write_text(str(model),encoding='utf-8')
    parameters=sum(p.numel() for p in model.parameters())
    convolutions=sum(isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)) for m in model.modules())
    print(f'Parameters: {parameters:,}; convolution modules including projections and gates: {convolutions}',flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5,betas=(.9,.999))
    best=-math.inf;stale=0
    with (out/'training.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['epoch','train_loss','validation_loss','validation_mIoU'])
        for epoch in range(1,args.epochs+1):
            model.train();total=0.;samples=0
            for images,masks,_ in loaders[0]:
                images,masks=images.to(args.device),masks.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss=hybrid_loss(model(images),masks)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();optimizer.step()
                total+=float(loss.detach())*len(images);samples+=len(images)
            validation,_=evaluate(model,loaders[1],args.device)
            writer.writerow([epoch,total/samples,validation['loss'],validation['mIoU']]);f.flush()
            print(f'Epoch {epoch}/{args.epochs}: train_loss={total/samples:.6f}, validation_loss={validation["loss"]:.6f}, validation_mIoU={validation["mIoU"]:.6f}',flush=True)
            if validation['mIoU']>best:
                best=validation['mIoU'];stale=0
                torch.save({'model':model.state_dict(),'base':args.base,'size':args.size,'classes':CLASSES,'epoch':epoch,'validation_mIoU':best},out/'best.pt')
            else:stale+=1
            if stale>=args.patience:break
    checkpoint=load_tensors(out/'best.pt',args.device);model.load_state_dict(checkpoint['model'])
    report,cm=evaluate(model,loaders[2],args.device,out/'predictions')
    report.update({'selected_epoch':checkpoint['epoch'],'test_images':len(split[2]),'parameters':parameters,'convolution_modules':convolutions,'demo_only':args.mode=='demo'})
    save_json(out/'metrics.json',report)
    with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['true_class / predicted_class']+CLASSES)
        for name,row in zip(CLASSES,cm.tolist()):w.writerow([name]+row)
    print(json.dumps(report,indent=2),flush=True)
    return report

def make_demo(folder):
    folder.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(42);records=[]
    for i in range(24):
        # Geometric label fields are software-test fixtures, not historical objects.
        mask=Image.new('L',(80,80),4);draw=ImageDraw.Draw(mask)
        for c in range(4):
            x=int(rng.integers(0,55));y=int(rng.integers(0,55))
            box=(x,y,min(79,x+int(rng.integers(15,35))),min(79,y+int(rng.integers(15,35))))
            if c%2:draw.rectangle(box,fill=c)
            else:draw.ellipse(box,fill=c)
        arr=COLORS[np.array(mask)].astype(float)+rng.normal(0,12,(80,80,3))
        im=folder/f'image_{i:03d}.png';ma=folder/f'mask_{i:03d}.png'
        Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).save(im);mask.save(ma)
        records.append({'image':str(im.resolve()),'mask':str(ma.resolve()),'group_id':f'synthetic_{i:03d}'})
    with (folder/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image','mask','group_id']);w.writeheader();w.writerows(records)
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',choices=['demo','train','cv','predict'],default='demo')
    p.add_argument('--manifest');p.add_argument('--checkpoint');p.add_argument('--image');p.add_argument('--encoder-weights')
    p.add_argument('--output',default='ra_unet_run');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--size',type=int);p.add_argument('--base',type=int);p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--patience',type=int,default=15);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();demo=args.mode=='demo'
    args.size=args.size if args.size is not None else (64 if demo else 512)
    args.base=args.base if args.base is not None else (8 if demo else 64)
    args.epochs=args.epochs if args.epochs is not None else (2 if demo else 200)
    args.batch_size=args.batch_size if args.batch_size is not None else (4 if demo else 8)
    if args.size<32 or args.size%16:p.error('--size must be a multiple of 16 and at least 32')
    if min(args.base,args.epochs,args.batch_size,args.patience,args.repeats,args.threads)<1 or args.lr<=0:p.error('Numeric settings must be positive')
    torch.set_num_threads(args.threads);seed_all(args.seed)
    out=Path(args.output)
    if out.exists():p.error('Output already exists. Choose a new --output directory.')
    if args.mode=='predict':
        if not args.checkpoint or not args.image:p.error('Prediction requires --checkpoint and --image')
        ck=load_tensors(args.checkpoint,args.device)
        if ck['classes']!=CLASSES:raise ValueError('Incompatible checkpoint class mapping')
        model=RAUNet(ck['base']).to(args.device);model.load_state_dict(ck['model']);model.eval()
        with Image.open(args.image) as original:
            with torch.no_grad():logits=model(image_tensor(preprocess(original,ck['size']))[None].to(args.device))[0]
            save_prediction(logits,original,out,Path(args.image).stem)
        save_json(out/'prediction_info.json',{'checkpoint':args.checkpoint,'image':args.image,'classes':CLASSES,'metrics':'No reference mask supplied; accuracy is not calculated.'})
        print('Prediction files saved to',out.resolve());return
    if not demo and not args.manifest:p.error('Training requires --manifest with annotated masks and source groups')
    if demo:
        out.mkdir(parents=True);rows=make_demo(out/'synthetic_demo_data')
        print('SYNTHETIC DEMO: execution check only; not paper accuracy.',flush=True)
        run_training(args,partition(rows,args.seed),out/'experiment',args.seed)
    else:
        rows=read_manifest(args.manifest)
        if args.mode=='train':run_training(args,partition(rows,args.seed),out,args.seed)
        else:
            out.mkdir(parents=True);fold_results=[];repeat_results=[]
            keys=['mIoU','macro_precision','macro_recall','macro_F1','pixel_accuracy']
            for repeat in range(args.repeats):
                current=[]
                for fold in range(5):
                    report=run_training(args,partition(rows,args.seed+repeat,fold),out/f'repeat_{repeat+1:02d}_fold_{fold+1}',args.seed+100*repeat+fold)
                    current.append(report);fold_results.append({'repeat':repeat+1,'fold':fold+1,**report})
                repeat_results.append({k:float(np.mean([r[k] for r in current if r[k] is not None])) for k in keys})
            summary={k:{'mean':float(np.mean([r[k] for r in repeat_results])),
                        'sample_sd_across_repeat_means':float(np.std([r[k] for r in repeat_results],ddof=1)) if args.repeats>1 else None} for k in keys}
            save_json(out/'cv_results.json',{'summary':summary,'repeat_means':repeat_results,'fold_results':fold_results,'note':'Unweighted fold means within each repeat. Repeated folds are not independent samples.'})
            print(json.dumps(summary,indent=2))
    print('Complete outputs:',out.resolve())

if __name__=='__main__':main()
