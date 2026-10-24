from pathlib import Path
import json,hashlib,csv,collections
import numpy as np
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated';OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Checksum mismatch: '+name
def bh(p):
 p=np.asarray(p);o=np.argsort(p);q=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out=np.empty_like(q);out[o]=np.minimum(q,1);return out
rr=ROOT/'results';rows=[];overlap=[]
for p in sorted(rr.glob('kablation_*.json')):
 if p.name=='kablation_cifar10_latest.json':continue
 x=json.loads(p.read_text())
 overlap.append(dict(file=p.name,records=len(x),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 if 'cifar' not in p.name:rows+=x
base={(x['k'],x['student'],x['seed']):x['f1_macro'] for x in rows if x['method']=='none'}
rows+=json.loads((rr/'aakd_benchmark.json').read_text())
methods=['akr','vanilla','takd','label_smoothing','ctkd','aakd','nkd','dist','dkd','simkd','crd','fitnet'];deltas={};tests=[]
for m in methods:
 ds={(x['k'],x['student'],x['seed']):x['f1_macro']-base[x['k'],x['student'],x['seed']] for x in rows if x['method']==m and x['params']<100000};assert len(ds)==280;deltas[m]=ds
 for k in sorted({key[0] for key in ds}):
  v=[v for key,v in ds.items() if key[0]==k];tests.append(dict(method=m,k=k,mean=float(np.mean(v)),p=float(wilcoxon(v).pvalue)))
for t,q in zip(tests,bh([t['p'] for t in tests])):t['q_bh84']=float(q)
comp=[]
for m in methods[1:]:
 v=[deltas['akr'][k]-v for k,v in deltas[m].items()];comp.append(dict(method=m,mean=float(np.mean(v)),p=float(wilcoxon(v).pvalue)))
for t,q in zip(comp,bh([t['p'] for t in comp])):t['q_bh11']=float(q)

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2))
def table(name,data):
    with (OUT/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
summary=[]
for m in methods:
    v=list(deltas[m].values());tt=[x for x in tests if x['method']==m]
    summary.append(dict(method=m,n=len(v),mean_delta_pp=float(np.mean(v)),sd=float(np.std(v,ddof=1)),wins=sum(x>0 for x in v),win_rate=100*np.mean(np.array(v)>0),sig_positive=sum(x['q_bh84']<.05 and x['mean']>0 for x in tt),sig_negative=sum(x['q_bh84']<.05 and x['mean']<0 for x in tt)))
configs=collections.defaultdict(list)
for (k,student,seed),v in deltas['akr'].items():configs[k,student].append(v)
assert sum(np.mean(v)>0 for v in configs.values())==21
assert summary[0]['wins']==167
assert abs(summary[0]['mean_delta_pp']-.6026460714285714)<1e-8
table('compact_comparison.csv',summary);table('method_task_bh84.csv',tests);table('akr_pairwise_bh11.csv',comp)
full={(x['k'],x['student'],x['seed']):x['f1_macro'] for x in rows if x['method']=='akr'}
vanilla={(x['k'],x['student'],x['seed']):x['f1_macro'] for x in rows if x['method']=='vanilla'}
factor=json.loads((rr/'factorial_results.json').read_text());fm={}
for x in factor:fm.setdefault(x['method'],{})[x['k'],x['student'],x['seed']]=x['f1_macro']
nota=json.loads((rr/'nota_complete/akr_nota_merged.json').read_text());assert len(nota)==80
keys=[(x['k'],x['student'],x['seed']) for x in nota];assert len(set(keys))==80
worker=[]
for p in (rr/'nota_complete').glob('akr_nota_results_*.json'):worker+=json.loads(p.read_text())
assert all(x in worker for x in nota), 'Merged record missing exact worker source'
fx=[]
for k in [28,7,6,5]:
    kk=[key for key in keys if key[0]==k];assert len(kk)==20
    n=[x['f1_macro'] for x in nota if x['k']==k]
    fx.append(dict(k=k,vanilla=np.mean([vanilla[z] for z in kk]),noskip=np.mean([fm['akr_noskip'][z] for z in kk]),skip_only=np.mean([fm['skip_only'][z] for z in kk]),no_ta=np.mean(n),full_akr=np.mean([full[z] for z in kk]),interaction=np.mean([full[z]-fm['akr_noskip'][z]-fm['skip_only'][z]+vanilla[z] for z in kk])))
table('factorial_and_nota.csv',fx)
d=[full[z]-x['f1_macro'] for z,x in zip(keys,nota)]
dump('nota_summary.json',dict(n=80,mean_full_minus_nota_pp=float(np.mean(d)),nominal_wilcoxon_p=float(wilcoxon(d).pvalue),source='results/nota_complete/akr_nota_merged.json',selection='Pre-existing merged archive; exact record matches to worker outputs; alternative attempts retained, not pooled.'))
sens=[]
for k in [6,28]:
 for student in ['micro','small']:
  for typ in ['threshold','alpha','label_smoothing']:
   p=rr/f'sensitivity_{typ}_k{k}_{student}.json';data=json.loads(p.read_text());groups=collections.defaultdict(list)
   for x in data:groups[x['config']].append(x['f1_macro'])
   for cfg,v in groups.items():sens.append(dict(file=p.name,config=cfg,n=len(v),mean_f1=float(np.mean(v)),sd=float(np.std(v,ddof=1))))
assert sum(x['n'] for x in sens)==600
table('sensitivity.csv',sens)
cif=json.loads((rr/'kablation_cifar10.json').read_text());cb={(x['student'],x['seed']):x['f1_macro'] for x in cif if x['method']=='none'}
cv=[x['f1_macro']-cb[x['student'],x['seed']] for x in cif if x['method']=='akr'];assert len(cv)==35
dump('summary.json',dict(artifact_checksums='PASS',compact_n=280,run_wins=167,configuration_wins=21,configurations=28,mean_delta_pp=summary[0]['mean_delta_pp'],factorial_interaction_pp=float(np.mean([x['interaction'] for x in fx])),cifar_n=len(cv),cifar_mean_delta_pp=float(np.mean(cv)),sensitivity_records=600,nota_records=80))
print('PASS: checksums, matched comparisons, factorial, 80 no-TA records, sensitivity and CIFAR summaries. Outputs: generated/')
