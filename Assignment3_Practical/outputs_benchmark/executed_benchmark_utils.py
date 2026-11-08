"""Shared routines for the executed project-held-out benchmark extension."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parent
FEATURES='wmc dit noc cbo rfc lcom ca ce npm lcom3 loc dam moa mfa cam ic cbm amc max_cc avg_cc'.split()
CK='wmc dit noc cbo rfc lcom'.split()
GROUPS={
 'size_and_methods':['wmc','npm','loc','amc'],
 'coupling_and_response':['cbo','rfc','ca','ce','ic','cbm'],
 'inheritance':['dit','noc','mfa'],
 'cohesion':['lcom','lcom3','cam'],
 'cyclomatic_complexity':['max_cc','avg_cc'],
 'access_and_aggregation':['dam','moa']}
CONDITIONS=[f'{name}_{subset}' for subset in ['all20','CK6'] for name in ['LR','RF','XGB']]
PARAMETERS={'LR':[1.0,0.1,10.0],'RF':[2,1,5],'XGB':[3,2,5]}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_table(path,key):
 df=pd.read_csv(path)
 required=set(FEATURES+['bug','name.1'])
 if not required.issubset(df.columns):raise ValueError(f'{key}: missing columns {required-set(df.columns)}')
 raw=len(df);labels=pd.to_numeric(df.bug,errors='coerce')
 valid=labels.notna() & (labels>=0) & (labels==labels.round()) & df['name.1'].notna()
 invalid=int((~valid).sum());df=df.loc[valid].copy()
 df['defective']=(labels.loc[valid]>0).astype(int)
 df['bug_count']=labels.loc[valid].astype(int)
 df['class_id']=df['name.1'].map(lambda x:hashlib.sha256((key+'|'+x).encode()).hexdigest()[:20])
 conflict=df.groupby('class_id').defective.transform('nunique')>1
 conflict_n=int(conflict.sum());df=df.loc[~conflict].copy()
 duplicate_n=int(df.duplicated('class_id').sum());df=df.drop_duplicates('class_id').copy()
 for col in FEATURES:df[col]=pd.to_numeric(df[col],errors='coerce')
 nonfinite=int(np.isinf(df[FEATURES].to_numpy()).sum())
 negative=int((df[FEATURES]<0).sum().sum())
 df[FEATURES]=df[FEATURES].replace([np.inf,-np.inf],np.nan).mask(df[FEATURES]<0)
 entirely_missing=int(df[FEATURES].isna().all(axis=1).sum())
 df=df.loc[~df[FEATURES].isna().all(axis=1)].copy()
 df['project']=key
 df['signature']=pd.util.hash_pandas_object(df[FEATURES],index=False).astype(str)
 audit={'project':key,'source_rows':raw,'invalid_label_or_id':invalid,'conflicting_id_rows':conflict_n,
        'duplicate_id_rows':duplicate_n,'entirely_missing_rows':entirely_missing,
        'negative_cells':negative,'nonfinite_cells':nonfinite,
        'missing_cells_retained':int(df[FEATURES].isna().sum().sum()),'clean_rows':len(df),
        'positive':int(df.defective.sum()),'negative':int((df.defective==0).sum()),
        'positive_rate':float(df.defective.mean()),'sha256':sha(path)}
 return df[['project','class_id','signature','bug_count','defective']+FEATURES].reset_index(drop=True),audit

def load_development():
 manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
 frames=[];audits=[]
 for row in manifest['files']:
  path=ROOT/'data'/f'{row["id"]}.csv'
  if sha(path)!=row['sha256']:raise ValueError(f'Input checksum mismatch {path}')
  frame,audit=load_table(path,row['id']);frames.append(frame);audits.append(audit)
 return pd.concat(frames,ignore_index=True),pd.DataFrame(audits)

def columns(condition):return FEATURES if condition.endswith('_all20') or condition=='Dummy' else CK

def build(condition,param,seed,training_labels):
 family=condition.split('_')[0]
 if family=='Dummy':model=DummyClassifier(strategy='prior')
 elif family=='LR':model=LogisticRegression(C=float(param),solver='lbfgs',max_iter=2000,class_weight='balanced',random_state=seed)
 elif family=='RF':model=RandomForestClassifier(n_estimators=200,min_samples_leaf=int(param),max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=1)
 elif family=='XGB':
  positives=int(np.sum(np.asarray(training_labels)==1));negatives=len(training_labels)-positives
  model=XGBClassifier(n_estimators=200,max_depth=int(param),learning_rate=.05,subsample=1,colsample_bytree=1,
                      scale_pos_weight=negatives/positives,tree_method='hist',eval_metric='logloss',random_state=seed,n_jobs=1)
 else:raise ValueError(condition)
 steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
 if family=='LR':steps.append(('scaler',StandardScaler()))
 return Pipeline(steps+[('model',model)])

def metric(y,p,threshold=.5):
 y=np.asarray(y);p=np.asarray(p);pred=(p>=threshold).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
 return {'n':len(y),'positives':int(y.sum()),'prevalence':float(y.mean()),'threshold':float(threshold),
         'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),
         'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),
         'roc_auc':float(auc),'average_precision':float(average_precision_score(y,p)),
         'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def tune_threshold(oof):
 candidates=[]
 for threshold in np.arange(.05,.951,.05):
  scores=[f1_score(g.actual,(g.probability>=threshold).astype(int),zero_division=0) for _,g in oof.groupby('validation_project')]
  candidates.append((float(np.mean(scores)),-abs(float(threshold)-.5),float(threshold)))
 best=max(candidates)
 return best[2],best[0]

def purge(evaluation,training):return evaluation.loc[~evaluation.signature.isin(set(training.signature))].copy()

def verify_pipeline(model,train,cols):
 medians=train[cols].median().to_numpy()
 if not np.allclose(model.named_steps['imputer'].statistics_,medians,equal_nan=True):raise AssertionError('Imputation is not training-only')
 if 'scaler' in model.named_steps:
  x=model.named_steps['imputer'].transform(train[cols])
  if not np.allclose(model.named_steps['scaler'].mean_,x.mean(axis=0)):raise AssertionError('Scaling is not training-only')

def permutation_rows(model,validation,seed,context):
 """Single and predefined group permutation; same permutation within a group."""
 baseline=average_precision_score(validation.defective,model.predict_proba(validation[FEATURES])[:,1])
 rng=np.random.default_rng(seed);base=validation[FEATURES].to_numpy()
 rows=[]
 for kind,groups in [('feature',{c:[c] for c in FEATURES}),('group',GROUPS)]:
  for name,members in groups.items():
   indices=[FEATURES.index(c) for c in members];drops=[]
   # One batched predict per feature/group reduces Python overhead.
   batches=[]
   for repeat in range(20):
    changed=base.copy();permutation=rng.permutation(len(base))
    changed[:,indices]=base[permutation][:,indices];batches.append(changed)
   all_prob=model.predict_proba(pd.DataFrame(np.concatenate(batches),columns=FEATURES))[:,1]
   for repeat in range(20):
    prob=all_prob[repeat*len(base):(repeat+1)*len(base)]
    drops.append(baseline-average_precision_score(validation.defective,prob))
   rows.append({**context,'kind':kind,'feature':name,'mean_ap_decrease':float(np.mean(drops)),
                'permutation_sd':float(np.std(drops)),'validation_ap':float(baseline),'repeats':20})
 return rows
