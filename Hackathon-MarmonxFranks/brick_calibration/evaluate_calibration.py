#!/usr/bin/env python3
"""Brick color calibration evaluator. Images must show ONE brick per image.

Folder structure:
 data/good/b01_good/*.png  ... data/bad/b05_bad/*.png
Optional surface masks: --masks masks; same relative path as image, PNG with >0 for brick surface.
If masks absent, a centered ROI is used (ONLY if it contains brick material exclusively).
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

EXT={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
FLAT_BRICK_DIR = re.compile(r'^Brick_(\d+)_(good|bad)$', re.IGNORECASE)

def write_csv(path, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def dE(a,b):
    return float(deltaE_ciede2000(np.asarray(a,dtype=float).reshape(1,1,3),np.asarray(b,dtype=float).reshape(1,1,3))[0,0])

def discover_bricks(data_root):
    """Return (directory, label) pairs for either supported dataset layout.

    Accepted layouts are the documented ``good/brick_id`` and
    ``bad/brick_id`` tree, or the captured-data layout with folders such as
    ``Brick_1_good`` directly below the dataset root.  In both cases the
    physical brick directory, rather than an individual image, is the held-out
    cross-validation group.
    """
    nested = []
    for label in ('good', 'bad'):
        label_root = data_root / label
        if label_root.is_dir():
            nested.extend((path, label) for path in label_root.iterdir()
                          if path.is_dir())
    flat = []
    for path in data_root.iterdir() if data_root.is_dir() else ():
        if not path.is_dir():
            continue
        match = FLAT_BRICK_DIR.fullmatch(path.name)
        if match:
            flat.append((path, match.group(2).lower()))
    if nested and flat:
        raise ValueError('Dataset mixes nested good/bad and flat Brick_N_label layouts')
    bricks = nested or flat
    if not bricks:
        raise ValueError(
            'No brick folders found; expected good/brick_id, bad/brick_id, '
            'or flat Brick_N_good / Brick_N_bad folders')
    return sorted(bricks, key=lambda item: item[0].name.lower())

def load_surface(image_path,data_root,masks_root,roi_fraction):
    bgr=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError('Could not decode image')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    if masks_root is not None:
        rel=image_path.relative_to(data_root)
        mp=(masks_root/rel).with_suffix('.png')
        mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
        if mask is None: raise ValueError(f'Missing mask: {mp}')
        if mask.shape!=(h,w): raise ValueError('Mask and image dimensions differ')
        mask=(mask>127).astype(np.uint8)
        mask=cv2.erode(mask,np.ones((5,5),np.uint8),iterations=1)
    else:
        # This is NOT segmentation: entire ROI must be brick material only.
        mask=np.zeros((h,w),np.uint8)
        rw,rh=int(w*roi_fraction),int(h*roi_fraction)
        mask[(h-rh)//2:(h+rh)//2,(w-rw)//2:(w+rw)//2]=1
    # Exclude nearly black shadow pixels and nearly white/specular clipped pixels.
    # This is a conservative heuristic, not a replacement for a material mask.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    value=hsv[:,:,2]
    usable=mask.astype(bool)&(value>15)&(value<250)
    if usable.sum()<150:raise ValueError('Too few usable brick pixels; check ROI/mask')
    lab=rgb2lab(rgb)
    p=lab[usable]
    med=np.median(p,axis=0)
    mean=np.mean(p,axis=0)
    std=np.std(p,axis=0)
    q10=np.percentile(p,10,axis=0);q90=np.percentile(p,90,axis=0)
    # Exposure QC on material mask before clipping exclusion.
    v=value[mask>0]
    return dict(median=med,features=np.r_[med,std,q10,q90],
                mean=mean,std=std,q10=q10,q90=q90,
                dark_pct=float(100*np.mean(v<=15)),
                clipped_pct=float(100*np.mean(v>=250)),
                median_brightness=float(np.median(v)),
                usable_fraction=float(usable.sum()/max(1,(mask>0).sum())))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True,help='Data directory with good/bad/brick_id/images')
    ap.add_argument('--out',type=Path,default=Path('results'))
    ap.add_argument('--masks',type=Path,help='Parallel tree of white-on-black surface masks')
    ap.add_argument('--roi-fraction',type=float,default=0.40,help='Centered square-fraction crop if masks omitted')
    args=ap.parse_args()
    if not 0<args.roi_fraction<=1: ap.error('--roi-fraction must be in (0,1]')
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[];Xs=[];ys=[];groups=[];meds=[]
    errors=[]
    try: bricks=discover_bricks(args.data)
    except ValueError as error: ap.error(str(error))
    for brickdir,label_dir in bricks:
        label=0 if label_dir=='good' else 1
        paths=sorted(p for p in brickdir.rglob('*') if p.is_file() and p.suffix.lower() in EXT)
        for path in paths:
            try: r=load_surface(path,args.data,args.masks,args.roi_fraction)
            except Exception as e:
                errors.append(f'{path}: {e}');continue
            rows.append(dict(brick_id=brickdir.name,label=label_dir,file=str(path),
                L_median=r['median'][0],a_median=r['median'][1],b_median=r['median'][2],
                L_std=r['std'][0],a_std=r['std'][1],b_std=r['std'][2],
                dark_pct=r['dark_pct'],clipped_pct=r['clipped_pct'],
                median_brightness=r['median_brightness'],usable_fraction=r['usable_fraction']))
            Xs.append(r['features']);ys.append(label);groups.append(brickdir.name);meds.append(r['median'])
    if not rows:ap.error('No valid images; check file structure and masks/ROI')
    write_csv(args.out/'image_quality_and_features.csv',rows)
    X=np.asarray(Xs);y=np.asarray(ys);g=np.asarray(groups);m=np.asarray(meds)
    unique=list(dict.fromkeys(g))
    brick_reps={b:np.median(m[g==b],axis=0) for b in unique}
    brick_labels={b:int(y[np.flatnonzero(g==b)[0]]) for b in unique}
    # Test A: within-brick color consistency.
    consistency=[]
    for b in unique:
        ds=np.asarray([dE(q,brick_reps[b]) for q in m[g==b]])
        consistency.append(dict(brick_id=b,label='bad' if brick_labels[b] else 'good',
           n_images=int(np.sum(g==b)),mean_deltaE00=float(ds.mean()),
           max_deltaE00=float(ds.max()),p90_deltaE00=float(np.percentile(ds,90))))
    write_csv(args.out/'within_brick_consistency.csv',consistency)
    # Test B: cross-brick color differences and nearest opposite-label brick.
    n=len(unique);dist=np.zeros((n,n));pairs=[]
    for i,a in enumerate(unique):
        for j,b in enumerate(unique):dist[i,j]=dE(brick_reps[a],brick_reps[b])
    for i,a in enumerate(unique):
        for j in range(i+1,n):
            b=unique[j]
            pairs.append(dict(brick_a=a,brick_b=b,relationship='opposite_label' if brick_labels[a]!=brick_labels[b] else 'same_label',deltaE00=float(dist[i,j])))
    write_csv(args.out/'pairwise_brick_distances.csv',pairs)
    nearest=[]
    for i,b in enumerate(unique):
        ix=[j for j,c in enumerate(unique) if brick_labels[b]!=brick_labels[c]]
        if ix:
            j=min(ix,key=lambda j:dist[i,j]);within=float(next(r['p90_deltaE00'] for r in consistency if r['brick_id']==b))
            nearest.append(dict(brick_id=b,label='bad' if brick_labels[b] else 'good',
                nearest_opposite_brick=unique[j],opposite_deltaE00=float(dist[i,j]),
                within_p90_deltaE00=within,separation_ratio=float(dist[i,j]/max(within,0.2))))
    write_csv(args.out/'nearest_opposite_class.csv',nearest)
    fig,ax=plt.subplots(figsize=(max(6,n*.65),max(5,n*.57)))
    im=ax.imshow(dist,cmap='viridis');ax.set_xticks(range(n),unique,rotation=65,ha='right');ax.set_yticks(range(n),unique)
    fig.colorbar(im,ax=ax,label='ΔE00');fig.tight_layout();fig.savefig(args.out/'color_distance_heatmap.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for b in unique:
        color='tab:red' if brick_labels[b] else 'tab:green'
        ax.scatter(m[g==b,1],m[g==b,2],s=23,c=color,alpha=.38)
        ax.scatter([brick_reps[b][1]],[brick_reps[b][2]],c=color,s=85,edgecolors='black')
        ax.annotate(b,(brick_reps[b][1],brick_reps[b][2]),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('a*');ax.set_ylabel('b*');ax.set_title('LAB chromaticity: good=green, bad=red')
    fig.tight_layout();fig.savefig(args.out/'lab_chromaticity.png',dpi=170);plt.close(fig)
    # Test C: group-held-out binary classification. No same brick in train/test.
    preds=np.full(len(y),-1);scores=np.full(len(y),np.nan)
    if len(unique)>=3 and len(set(y))==2:
        for train,test in LeaveOneGroupOut().split(X,y,g):
            if len(np.unique(y[train]))<2:continue
            model=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced'))
            model.fit(X[train],y[train]);preds[test]=model.predict(X[test]);scores[test]=model.predict_proba(X[test])[:,1]
    valid=(preds!=-1)
    predictions=[];brick_predictions=[]
    for k,r in enumerate(rows):
        predictions.append(dict(brick_id=r['brick_id'],file=r['file'],true_label=r['label'],
            predicted_label=('bad' if preds[k]==1 else 'good') if valid[k] else 'not_evaluated',
            bad_probability=float(scores[k]) if valid[k] else ''))
    write_csv(args.out/'held_out_image_predictions.csv',predictions)
    for b in unique:
        ix=np.flatnonzero((g==b)&valid)
        if not len(ix):continue
        avg=float(np.mean(scores[ix]));pred=int(avg>=0.5);true=brick_labels[b]
        brick_predictions.append(dict(brick_id=b,true_label='bad' if true else 'good',
            predicted_label='bad' if pred else 'good',mean_bad_probability=avg,
            correct=int(true==pred),n_images=len(ix)))
    write_csv(args.out/'held_out_brick_predictions.csv',brick_predictions)
    summary={'n_images':len(rows),'n_bricks':len(unique),'n_good_bricks':sum(v==0 for v in brick_labels.values()),
        'n_bad_bricks':sum(v==1 for v in brick_labels.values()),'mean_within_brick_deltaE00':float(np.mean([r['mean_deltaE00'] for r in consistency])),
        'image_clipping_flags':sum(r['clipped_pct']>1 for r in rows),
        'roi_warning':args.masks is None,
        'note':'Thresholds/ratios are descriptive diagnostics, not validated pass/fail criteria. Same-brick photos are correlated.'}
    if nearest:summary['minimum_between_opposite_labels_deltaE00']=float(min(r['opposite_deltaE00'] for r in nearest))
    if brick_predictions:
        t=np.array([int(r['true_label']=='bad') for r in brick_predictions]);p=np.array([int(r['predicted_label']=='bad') for r in brick_predictions])
        summary['held_out_brick_metrics']={
            'accuracy':float(accuracy_score(t,p)),
            'bad_precision':float(precision_score(t,p,zero_division=0)),
            'bad_recall':float(recall_score(t,p,zero_division=0)),
            'bad_f1':float(f1_score(t,p,zero_division=0)),
            'confusion_matrix_rows_good_bad_cols_good_bad':confusion_matrix(t,p,labels=[0,1]).tolist()}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    if errors:(args.out/'skipped_images.txt').write_text('\n'.join(errors))
    print(json.dumps(summary,indent=2))
    print(f'Outputs: {args.out.resolve()}')
    if errors:print(f'WARNING: {len(errors)} unreadable / mask-invalid images; see skipped_images.txt')
    if args.masks is None:print('WARNING: Center ROI only. Verify that ROI contains brick MATERIAL, not holes/background/shadows!')

if __name__=='__main__':main()
