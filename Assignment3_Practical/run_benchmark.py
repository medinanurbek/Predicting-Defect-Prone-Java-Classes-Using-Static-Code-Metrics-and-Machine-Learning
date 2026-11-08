"""Execute nested leave-one-project-out development evaluation.

Never reads the new Synapse holdout. The original pilot and outputs remain intact.
Primary: AP selection and threshold 0.5. Secondary: inner-validation macro-F1 threshold.
"""
import argparse
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import sklearn,scipy,xgboost
from sklearn.metrics import average_precision_score
from benchmark_utils import ROOT,FEATURES,CK,GROUPS,CONDITIONS,PARAMETERS,sha,load_development,columns,build,metric,tune_threshold,purge,verify_pipeline,permutation_rows

SCORES=['accuracy','precision','recall','f1','roc_auc','average_precision']

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs_benchmark');parser.add_argument('--reproduce',action='store_true',help='Recompute development folds after the holdout has already been released; makes no fresh-holdout claim.');args=parser.parse_args()
 out=ROOT/args.output
 if out.exists() and any(out.iterdir()):raise RuntimeError('Output directory is not empty. Use a different --output directory to preserve earlier evidence.')
 out.mkdir(exist_ok=True,parents=True)
 data,inventory=load_development();projects=sorted(data.project.unique());seeds=[42,43,44]
 holdout_present=(ROOT/'data_holdout/synapse-1.2.csv').exists()
 protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'design':'nested leave-one-project-out',
           'development_projects':projects,'seeds':seeds,'primary_threshold':.5,
           'selection':'equal-project mean inner validation average precision',
           'secondary_threshold':'maximise equal-project inner validation F1 on grid 0.05 to 0.95 by 0.05; ties closest to 0.5',
           'parameter_grid':PARAMETERS,'conditions':CONDITIONS,'features':FEATURES,'ck_features':CK,
           'single_and_joint_permutation_groups':GROUPS,'importance_seed':42,'permutation_repeats':20,
           'overlap_rule':'exclude evaluation vectors that exactly match its training vectors',
           'sensitivity':'evaluate all outer test records with the same fitted model and threshold',
           'future_holdout_project':'synapse-1.2','future_holdout_member':'UnifiedBugDataset-1.2/PROMISE/orig/defect/ck/synapse/synapse-1.2.csv',
           'holdout_rule':'initial run: do not acquire or inspect holdout before final_selection.json is written; reproduction: released holdout is never read by this program',
           'reproduction_run':bool(args.reproduce),'holdout_absent_at_start':not holdout_present,
           'source_sha256':dict(zip(inventory.project,inventory.sha256)),
           'script_sha256':sha(__file__),'utils_sha256':sha(ROOT/'benchmark_utils.py')}
 if holdout_present and not args.reproduce:raise RuntimeError('Holdout already exists. Use --reproduce to recompute without claiming fresh confirmation.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 (out/'executed_run_benchmark.py').write_bytes(Path(__file__).read_bytes())
 (out/'executed_benchmark_utils.py').write_bytes((ROOT/'benchmark_utils.py').read_bytes())
 inventory.to_csv(out/'data_inventory.csv',index=False);data.to_csv(out/'analysis_ready_development.csv',index=False)
 inventory[['project','clean_rows','positive','negative','positive_rate']].to_csv(out/'exploratory_prevalence.csv',index=False)
 distributions=[]
 for project,g in data.groupby('project'):
  for feature in FEATURES:
   x=g[feature]
   distributions.append({'project':project,'feature':feature,'median':float(x.median()),'q1':float(x.quantile(.25)),
                         'q3':float(x.quantile(.75)),'minimum':float(x.min()),'maximum':float(x.max()),
                         'zero_rate':float((x==0).mean())})
 pd.DataFrame(distributions).to_csv(out/'exploratory_distributions.csv',index=False)
 correlations=[]
 for project,g in data.groupby('project'):
  corr=g[FEATURES].corr(method='spearman')
  for i,left in enumerate(FEATURES):
   for right in FEATURES[i+1:]:correlations.append({'project':project,'feature_a':left,'feature_b':right,'spearman':corr.loc[left,right]})
 pd.DataFrame(correlations).to_csv(out/'exploratory_correlations.csv',index=False)
 all_inner=[];all_results=[];all_predictions=[];all_sensitivity=[];all_importance=[];all_folds=[];events=[]
 pred_path=out/'inner_validation_predictions.csv.gz'
 def log(s):
  print(s,flush=True);events.append(s);(out/'run.log').write_text('\n'.join(events)+'\n',encoding='utf-8')
 log(f'Protocol frozen before fitting: {len(projects)} projects, {len(data)} records, three seeds.')
 log('The additional Synapse project is not loaded by this program.')
 total_contexts=len(projects)*len(seeds);done=0;pipeline_checks=0
 for seed in seeds:
  for test_project in projects:
   train=data.loc[data.project!=test_project].copy();test_all=data.loc[data.project==test_project].copy();test=purge(test_all,train)
   if set(train.project)&set(test.project):raise AssertionError('Outer project leakage')
   if set(train.signature)&set(test.signature):raise AssertionError('Outer exact-vector overlap')
   if test.defective.nunique()!=2:raise ValueError('Undefined ROC-AUC in outer test')
   all_folds.append({'seed':seed,'test_project':test_project,'train_rows':len(train),'test_before':len(test_all),
                     'test_after':len(test),'test_excluded':len(test_all)-len(test),'test_positive':int(test.defective.sum())})
   log(f'Outer {done+1}/{total_contexts}: seed={seed}, test={test_project}, train={len(train)}, retained test={len(test)}')
   context_predictions=[];context_scores=[]
   for condition in CONDITIONS:
    cols=columns(condition);family=condition.split('_')[0]
    for param in PARAMETERS[family]:
     for val_project in sorted(train.project.unique()):
      inner_train=train.loc[train.project!=val_project].copy();val_all=train.loc[train.project==val_project].copy();val=purge(val_all,inner_train)
      if set(inner_train.project)&set(val.project):raise AssertionError('Inner project leakage')
      if set(inner_train.signature)&set(val.signature):raise AssertionError('Inner exact-vector overlap')
      model=build(condition,param,seed,inner_train.defective).fit(inner_train[cols],inner_train.defective)
      verify_pipeline(model,inner_train,cols);pipeline_checks+=1
      prob=model.predict_proba(val[cols])[:,1]
      row={'seed':seed,'test_project':test_project,'validation_project':val_project,'condition':condition,
           'parameter':param,'validation_n':len(val),'validation_positive':int(val.defective.sum()),
           'excluded_validation':len(val_all)-len(val),'average_precision':float(average_precision_score(val.defective,prob))}
      context_scores.append(row)
      context_predictions.append(pd.DataFrame({'seed':seed,'outer_test_project':test_project,'validation_project':val_project,
                           'condition':condition,'parameter':param,'class_id':val.class_id.to_numpy(),
                           'actual':val.defective.to_numpy(),'probability':prob}))
   inner=pd.DataFrame(context_scores);all_inner.extend(context_scores)
   oof=pd.concat(context_predictions,ignore_index=True)
   oof.to_csv(pred_path,index=False,mode='a' if pred_path.exists() else 'w',header=not pred_path.exists(),compression='gzip')
   mean_ap=inner.groupby(['condition','parameter'],sort=False).average_precision.mean().reset_index()
   best_rows=[]
   for condition in CONDITIONS:
    choices=mean_ap.loc[mean_ap.condition==condition];best=choices.loc[choices.average_precision.idxmax()];best_rows.append(best.to_dict())
    param=float(best.parameter);cols=columns(condition)
    chosen_oof=oof.loc[(oof.condition==condition)&np.isclose(oof.parameter,param)]
    threshold,inner_f1=tune_threshold(chosen_oof)
    model=build(condition,param,seed,train.defective).fit(train[cols],train.defective)
    verify_pipeline(model,train,cols);pipeline_checks+=1
    prob=model.predict_proba(test[cols])[:,1];prob_all=model.predict_proba(test_all[cols])[:,1]
    for mode,t in [('fixed_0.5',.5),('validation_F1',threshold)]:
     result={'seed':seed,'test_project':test_project,'condition':condition,'parameter':param,
             'threshold_policy':mode,'inner_ap':float(best.average_precision),'inner_threshold_f1':inner_f1,**metric(test.defective,prob,t)}
     all_results.append(result)
     all_sensitivity.append({'seed':seed,'test_project':test_project,'condition':condition,'parameter':param,
                             'threshold_policy':mode,'excluded_in_primary':len(test_all)-len(test),**metric(test_all.defective,prob_all,t)})
    all_predictions.append(pd.DataFrame({'seed':seed,'test_project':test_project,'condition':condition,'parameter':param,
                           'class_id':test.class_id.to_numpy(),'actual':test.defective.to_numpy(),'probability':prob,
                           'fixed_prediction':(prob>=.5).astype(int),'validation_threshold':threshold,'validation_prediction':(prob>=threshold).astype(int)}))
   # Trivial references make an indiscriminate high-recall result visible.
   training_prior=float(train.defective.mean())
   for name,reference_probability in [('Dummy',training_prior),('AlwaysPositive',1.0)]:
    all_results.append({'seed':seed,'test_project':test_project,'condition':name,'parameter':None,
                        'threshold_policy':'fixed_0.5','inner_ap':None,'inner_threshold_f1':None,
                        **metric(test.defective,np.full(len(test),reference_probability),.5)})
   if seed==42:
    full=[row for row in best_rows if row['condition'].endswith('_all20')]
    selected=max(full,key=lambda row:row['average_precision'])
    for val_project in sorted(train.project.unique()):
     inner_train=train.loc[train.project!=val_project];val=purge(train.loc[train.project==val_project],inner_train)
     model=build(selected['condition'],selected['parameter'],seed,inner_train.defective).fit(inner_train[FEATURES],inner_train.defective)
     all_importance.extend(permutation_rows(model,val,seed,{'outer_test_project':test_project,'validation_project':val_project,
                                   'condition':selected['condition'],'parameter':selected['parameter'],'seed':seed}))
   done+=1
   pd.DataFrame(all_results).to_csv(out/'outer_metrics.csv',index=False)
   pd.DataFrame(all_inner).to_csv(out/'inner_cv_scores.csv',index=False)
   log(f'Completed {done}/{total_contexts}; validation choices fixed before scoring each outer test.')
 results=pd.DataFrame(all_results);sensitivity=pd.DataFrame(all_sensitivity);importance=pd.DataFrame(all_importance)
 sensitivity.to_csv(out/'metrics_including_matching_test_vectors.csv',index=False)
 pd.concat(all_predictions,ignore_index=True).to_csv(out/'outer_predictions.csv.gz',index=False,compression='gzip')
 pd.DataFrame(all_folds).to_csv(out/'fold_inventory.csv',index=False)
 importance.to_csv(out/'validation_importance_contexts.csv',index=False)
 importance['top5']=False
 for _,g in importance.loc[importance.kind=='feature'].groupby(['outer_test_project','validation_project']):
  importance.loc[g.nlargest(5,'mean_ap_decrease').index,'top5']=True
 importance.groupby(['kind','feature']).agg(mean_ap_decrease=('mean_ap_decrease','mean'),context_sd=('mean_ap_decrease','std'),
                minimum=('mean_ap_decrease','min'),maximum=('mean_ap_decrease','max'),contexts=('mean_ap_decrease','size'),
                top5_frequency=('top5','mean')).reset_index().sort_values(['kind','mean_ap_decrease'],ascending=[True,False]).to_csv(out/'importance_summary.csv',index=False)
 project=results.groupby(['condition','threshold_policy','test_project'])[SCORES].mean().reset_index()
 project.to_csv(out/'per_project_metrics.csv',index=False)
 macro=project.groupby(['condition','threshold_policy'])[SCORES].agg(['mean','std','min','max'])
 macro.columns=['_'.join(c) for c in macro.columns];macro.reset_index().to_csv(out/'macro_summary.csv',index=False)
 seed_variation=results.groupby(['condition','threshold_policy','test_project'])[SCORES].std().reset_index()
 seed_variation.to_csv(out/'seed_standard_deviations.csv',index=False)
 paired=project.loc[project.condition.isin(CONDITIONS)].merge(project.loc[project.condition=='Dummy',['test_project']+SCORES],on='test_project',suffixes=('','_dummy'))
 for score in SCORES:paired[score+'_minus_dummy']=paired[score]-paired[score+'_dummy']
 paired.to_csv(out/'paired_differences_from_dummy.csv',index=False)
 # Select the additional-project model using INNER validation scores only.
 inner=pd.DataFrame(all_inner)
 global_scores=inner.groupby(['condition','parameter','validation_project'],sort=False).average_precision.mean().reset_index()
 global_mean=global_scores.groupby(['condition','parameter'],sort=False).average_precision.mean().reset_index()
 global_mean.to_csv(out/'development_selection_scores.csv',index=False)
 best=global_mean.loc[global_mean.average_precision.idxmax()]
 kept=[]
 for chunk in pd.read_csv(pred_path,chunksize=50000):
  kept.append(chunk.loc[(chunk.condition==best.condition)&np.isclose(chunk.parameter,best.parameter)])
 final_oof=pd.concat(kept,ignore_index=True).groupby(['validation_project','class_id','actual']).probability.mean().reset_index()
 final_threshold,final_f1=tune_threshold(final_oof)
 final_oof.to_csv(out/'final_selection_oof_predictions.csv',index=False)
 final={'frozen_utc':datetime.now(timezone.utc).isoformat(),'condition':str(best.condition),'parameter':float(best.parameter),
        'mean_inner_project_ap':float(best.average_precision),'secondary_threshold':final_threshold,
        'secondary_threshold_inner_macro_f1':final_f1,'primary_threshold':.5,'seeds':seeds,
        'selection_data':'inner validation only; no Synapse data and no outer-test score used for final choice',
        'holdout_project':'synapse-1.2','protocol_sha256':sha(out/'protocol.json')}
 (out/'final_selection.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
 checks=[f'PASS SHA256: all five original source files match their manifests.',
         'PASS project separation and exact-vector exclusion in every inner and outer split.',
         f'PASS training-only imputation and scaling checks in {pipeline_checks} fitted pipelines.',
         'PASS model and threshold selection uses only inner validation predictions.',
         'PASS three seeds and five outer projects have complete primary and secondary scores.',
         'PASS additional holdout was never loaded by the development benchmark.' if args.reproduce else 'PASS additional holdout was not acquired or loaded before final selection.']
 assert len(results.loc[results.threshold_policy=='fixed_0.5'])==15*8
 assert len(results.loc[results.threshold_policy=='validation_F1'])==15*6
 assert all(np.isfinite(results[SCORES]).all())
 assert (results[['tn','fp','fn','tp']].sum(axis=1)==results.n).all()
 # Recompute every saved outer score from prediction rows.
 prediction_frame=pd.concat(all_predictions,ignore_index=True)
 for (seed,test_project,condition),g in prediction_frame.groupby(['seed','test_project','condition']):
  for mode,t in [('fixed_0.5',.5),('validation_F1',float(g.validation_threshold.iloc[0]))]:
   saved=results.loc[(results.seed==seed)&(results.test_project==test_project)&(results.condition==condition)&(results.threshold_policy==mode)].iloc[0]
   calculated=metric(g.actual,g.probability,t)
   if not all(np.isclose(saved[score],calculated[score]) for score in SCORES):raise AssertionError('Score mismatch')
 checks.append('PASS all primary and secondary model metrics recomputed from saved predictions.')
 (out/'checks.txt').write_text('\n'.join(checks)+'\n',encoding='utf-8')
 env={'executed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
      'versions':{m.__name__:m.__version__ for m in [np,pd,scipy,sklearn,xgboost]}}
 (out/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
 (out/'completed.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
                 'outer_project_seed_contexts':done,'outer_metric_rows':len(results),'inner_cv_fits':len(inner),
                 'protocol_sha256':sha(out/'protocol.json'),'final_selection_sha256':sha(out/'final_selection.json')},indent=2),encoding='utf-8')
 log(f'COMPLETE: {done} outer contexts; {len(inner)} inner fits; final holdout choice={best.condition}, parameter={best.parameter}, threshold={final_threshold:.2f}')

if __name__=='__main__':main()
