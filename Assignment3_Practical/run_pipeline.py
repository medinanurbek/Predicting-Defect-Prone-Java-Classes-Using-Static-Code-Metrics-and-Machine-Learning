"""Initial Java class defect benchmark. Run from any working directory.

Protocol fixed before evaluation: Ant/JEdit/POI train, Ivy validate, Camel test.
All configurations are predeclared. Validation AP selects one full-feature model;
test outcomes are never used to select, tune, or refit a model.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
FEATURES = 'wmc dit noc cbo rfc lcom ca ce npm lcom3 loc dam moa mfa cam ic cbm amc max_cc avg_cc'.split()
CK = 'wmc dit noc cbo rfc lcom'.split()
SPLITS = {'ant-1.7':'train', 'jedit-4.3':'train', 'poi-3.0':'train',
          'ivy-2.0':'validation', 'camel-1.6':'test'}
SEED = 42

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def measure(y, probability):
    predicted = (probability >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,predicted), 'precision':precision_score(y,predicted,zero_division=0),
            'recall':recall_score(y,predicted,zero_division=0), 'f1':f1_score(y,predicted,zero_division=0),
            'roc_auc':roc_auc_score(y,probability), 'average_precision':average_precision_score(y,probability),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True,exist_ok=True)
    events = []
    def log(message):
        print(message,flush=True)
        events.append(message)
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    inventory=[]
    frames=[]
    for entry in manifest['files']:
        key=entry['id']; path=ROOT/'data'/f'{key}.csv'
        assert sha(path)==entry['sha256'],f'Checksum mismatch: {path}'
        df=pd.read_csv(path)
        raw=len(df)
        assert set(FEATURES+['bug','name.1']).issubset(df.columns)
        labels=pd.to_numeric(df['bug'],errors='coerce')
        valid=labels.notna() & (labels>=0) & (labels==labels.round()) & df['name.1'].notna()
        df=df.loc[valid].copy()
        df['bug_count']=labels.loc[valid].astype(int)
        df['defective']=(df['bug_count']>0).astype(int)
        df['class_id']=df['name.1'].map(lambda v:hashlib.sha256((key+'|'+v).encode()).hexdigest()[:20])
        conflict=df.groupby('class_id')['defective'].transform('nunique')>1
        conflicts=int(conflict.sum())
        df=df.loc[~conflict].drop_duplicates('class_id',keep='first').copy()
        for col in FEATURES:df[col]=pd.to_numeric(df[col],errors='coerce')
        df[FEATURES]=df[FEATURES].replace([np.inf,-np.inf],np.nan)
        negative=int((df[FEATURES]<0).sum().sum())
        df[FEATURES]=df[FEATURES].mask(df[FEATURES]<0)
        df=df.loc[~df[FEATURES].isna().all(axis=1)].copy()
        df['project']=key
        df['split']=SPLITS[key]
        df['signature']=pd.util.hash_pandas_object(df[FEATURES],index=False).astype(str)
        inventory.append({'project':key,'split':SPLITS[key],'raw_rows':raw,'valid_rows':len(df),
                          'excluded_rows':raw-len(df),'conflicting_id_rows':conflicts,
                          'negative_cells_to_missing':negative,'missing_feature_cells':int(df[FEATURES].isna().sum().sum()),
                          'defective':int(df.defective.sum()),'non_defective':int((df.defective==0).sum())})
        frames.append(df[['project','class_id','signature','split','bug_count','defective']+FEATURES])
    combined=pd.concat(frames,ignore_index=True)
    # Purge exact feature vectors shared across splits. Distinct classes within
    # one split are retained; matching vectors do not prove identical code.
    train=combined.loc[combined.split=='train'].copy()
    val_all=combined.loc[combined.split=='validation'].copy()
    validation=val_all.loc[~val_all.signature.isin(set(train.signature))].copy()
    test_all=combined.loc[combined.split=='test'].copy()
    test=test_all.loc[~test_all.signature.isin(set(train.signature)|set(val_all.signature))].copy()
    parts={'train':train,'validation':validation,'test':test}
    audit={'features':FEATURES,'ck_features':CK,'seed':SEED,'threshold':.5,
           'validation_vectors_purged':len(val_all)-len(validation),
           'test_vectors_purged':len(test_all)-len(test),'splits':{}}
    for split,df in parts.items():
        audit['splits'][split]={'rows':len(df),'defective':int(df.defective.sum()),
                               'non_defective':int((df.defective==0).sum()),'projects':sorted(df.project.unique())}
        assert df.defective.nunique()==2
        log(f'{split}: n={len(df)}, positive={int(df.defective.sum())}, projects={sorted(df.project.unique())}')
    checks=[]
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for col in ['project','class_id','signature']:
            assert not (set(parts[a][col]) & set(parts[b][col])),f'{col} leakage'
        checks.append(f'PASS {a}/{b}: disjoint projects, class IDs and exact feature vectors')
    pd.DataFrame(inventory).to_csv(out/'dataset_summary.csv',index=False)
    processed=pd.concat(parts.values(),ignore_index=True)
    processed.to_csv(out/'processed_dataset.csv',index=False)
    processed[['project','class_id','signature','split','defective']].to_csv(out/'split_manifest.csv',index=False)
    train[FEATURES].describe().T.to_csv(out/'training_descriptive_statistics.csv')
    train[FEATURES].corr(method='spearman').to_csv(out/'training_spearman.csv')
    ratio=float((train.defective==0).sum()/train.defective.sum())
    def build(name):
        if name=='Dummy': estimator=DummyClassifier(strategy='prior')
        elif name=='LR': estimator=LogisticRegression(C=1,max_iter=2000,solver='lbfgs',class_weight='balanced',random_state=SEED)
        elif name=='RF': estimator=RandomForestClassifier(n_estimators=200,max_depth=None,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=1)
        else: estimator=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,subsample=1,colsample_bytree=1,scale_pos_weight=ratio,tree_method='hist',eval_metric='logloss',random_state=SEED,n_jobs=1)
        steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
        if name=='LR':steps.append(('scaler',StandardScaler()))
        return Pipeline(steps+[('model',estimator)])
    configurations=[('Dummy','all20'),('LR','all20'),('RF','all20'),('XGB','all20'),('LR','CK6'),('RF','CK6'),('XGB','CK6')]
    models={};validation_rows=[];configs={}
    for name,subset in configurations:
        key=f'{name}_{subset}'; cols=FEATURES if subset=='all20' else CK
        model=build(name).fit(train[cols],train.defective)
        models[key]=(model,cols)
        configs[key]={k:str(v) for k,v in model.get_params(deep=True).items() if k not in ['steps','model','imputer','scaler']}
        row={'model':key,'split':'validation',**measure(validation.defective,model.predict_proba(validation[cols])[:,1])}
        validation_rows.append(row)
        log(f'{key} validation AP={row["average_precision"]:.6f} F1={row["f1"]:.6f}')
    selection=max((r for r in validation_rows if r['model'] in ['LR_all20','RF_all20','XGB_all20']),key=lambda r:r['average_precision'])['model']
    log('Selected by validation average precision, before test evaluation: '+selection)
    model,cols=models[selection]
    importance=permutation_importance(model,validation[cols],validation.defective,n_repeats=20,random_state=SEED,scoring='average_precision',n_jobs=1)
    pd.DataFrame({'feature':cols,'mean_ap_decrease':importance.importances_mean,'repeat_sd':importance.importances_std}).sort_values('mean_ap_decrease',ascending=False).to_csv(out/'validation_permutation_importance.csv',index=False)
    rows=list(validation_rows);predictions=[]
    for key,(model,cols) in models.items():
        probability=model.predict_proba(test[cols])[:,1]
        row={'model':key,'split':'test',**measure(test.defective,probability)}
        rows.append(row)
        for idx,prob in zip(test.index,probability):
            predictions.append({'model':key,'project':test.loc[idx,'project'],'class_id':test.loc[idx,'class_id'],
                                'actual':int(test.loc[idx,'defective']),'probability':float(prob),'predicted':int(prob>=.5)})
        log(f'{key} test AP={row["average_precision"]:.6f} F1={row["f1"]:.6f} FN={row["fn"]}')
    pred=pd.DataFrame(predictions)
    pred.to_csv(out/'test_predictions.csv',index=False)
    errors=pred.loc[(pred.model==selection)&(pred.actual!=pred.predicted)].merge(test[['class_id']+FEATURES],on='class_id')
    errors.to_csv(out/'selected_model_errors.csv',index=False)
    checks.append('PASS all raw CSV SHA256 checks and ZIP CRC verification at download')
    for key,(model,cols) in models.items():
        assert np.allclose(model.named_steps['imputer'].statistics_,train[cols].median().to_numpy(),equal_nan=True)
    checks.append('PASS fitted imputation medians equal training medians for all seven configurations')
    lr=models['LR_all20'][0]
    imputed=lr.named_steps['imputer'].transform(train[FEATURES])
    assert np.allclose(lr.named_steps['scaler'].mean_,imputed.mean(axis=0))
    checks.append('PASS LR scaling means equal training-only imputed means')
    for row in rows:
        if row['split']=='test':
            p=pred.loc[pred.model==row['model']]
            check=measure(p.actual,p.probability)
            assert all(np.isclose(check[k],row[k]) for k in check)
            assert row['tn']+row['fp']+row['fn']+row['tp']==len(test)
    checks.append('PASS test metrics recomputed from saved prediction records and confusion-matrix totals')
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    audit.update({'selected_full_model':selection,'training_class_weight_ratio':ratio,
                  'raw_total':sum(r['raw_rows'] for r in inventory),'retained_total':len(processed),
                  'data_sha256':{e['id']:e['sha256'] for e in manifest['files']},
                  'script_sha256':sha(Path(__file__)),'checks_passed':len(checks),
                  'started_protocol':'fixed pilot, no tuning or post-test refitting'})
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (out/'configurations.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
    env={'executed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'os':platform.platform(),
         'processor':platform.processor(),'machine':platform.machine(),
         'libraries':{p.__name__:p.__version__ for p in [numpy_module for numpy_module in [np,pd,scipy,sklearn,xgboost]]}}
    (out/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    (out/'checks.txt').write_text('\n'.join(checks)+'\n',encoding='utf-8')
    for message in checks:log(message)
    (out/'run.log').write_text('\n'.join(events)+'\n',encoding='utf-8')
    log('Completed real-data initial implementation. No final multi-project benchmark is claimed.')

if __name__=='__main__':main()
