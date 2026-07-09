import sys, numpy as np
from sklearn.model_selection import train_test_split
ROOT=sys.argv[1]; NOISE=0.01; TEST=0.30; SEED=42
def _b1(rng,n,d):  # linear base + deg2 + deg3 (no parity)
    x=rng.standard_normal((n,d)); a=x[:,:10]
    lin=0.85*a[:,0]-0.75*a[:,1]+0.6*a[:,2]
    d2=1.3*a[:,3]*a[:,4]-1.2*a[:,5]*a[:,6]
    d3=1.4*a[:,7]*a[:,8]*a[:,9]
    m=lin+d2+d3; return x,(m>np.median(m)).astype(int)
def _b2(rng,n,d):  # base + deg2 + deg3 + one xor-of-2 (deg2 parity)
    x=rng.standard_normal((n,d)); a=x[:,:12]
    lin=0.8*a[:,0]-0.7*a[:,1]
    d2=1.3*a[:,2]*a[:,3]+1.2*a[:,4]*a[:,5]
    d3=1.4*a[:,6]*a[:,7]*a[:,8]-1.3*a[:,9]*a[:,10]*a[:,11]
    xor2=((a[:,0]>0).astype(int)^(a[:,2]>0).astype(int)).astype(float)
    m=lin+d2+d3+1.4*(xor2-0.5); return x,(m>np.median(m)).astype(int)
def _m3(rng,n,d):  # 3-class: base + deg2 + deg3
    x=rng.standard_normal((n,d)); a=x[:,:14]
    s=(0.7*a[:,0]-0.6*a[:,1]+1.3*a[:,2]*a[:,3]+1.2*a[:,4]*a[:,5]
       +1.4*a[:,6]*a[:,7]*a[:,8]-1.3*a[:,9]*a[:,10]*a[:,11])
    q=np.quantile(s,[1/3,2/3]); return x,np.digitize(s,q).astype(int)
specs=[(_b1,3200,50,11),(_b2,3200,55,23),(_m3,3400,60,37)]
out={}
for i,(b,n,d,sd) in enumerate(specs):
    rng=np.random.default_rng(sd); x,y=b(rng,n,d)
    flip=rng.random(len(y))<NOISE
    if flip.any(): y=y.copy(); y[flip]=rng.choice(np.unique(y),size=int(flip.sum()))
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=TEST,stratify=y,random_state=SEED)
    out[f"d{i}_xtr"]=xtr.astype(np.float64);out[f"d{i}_ytr"]=ytr.astype(np.int64)
    out[f"d{i}_xte"]=xte.astype(np.float64);out[f"d{i}_yte"]=yte.astype(np.int64)
out["n_datasets"]=np.array(3); np.savez_compressed(ROOT+"/tasks/tabular-blind/data/datasets.npz",**out)
print("ok",[out[f'd{i}_xtr'].shape for i in range(3)])
