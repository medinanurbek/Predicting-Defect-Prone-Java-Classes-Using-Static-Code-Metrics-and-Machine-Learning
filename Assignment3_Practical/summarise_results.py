"""Summarise completed, saved evidence without fitting any model."""
from pathlib import Path
import json
import argparse
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--benchmark-output',default='outputs_benchmark')
parser.add_argument('--holdout-output',default='outputs_holdout')
args=parser.parse_args()
B=ROOT/args.benchmark_output;H=ROOT/args.holdout_output
macro=pd.read_csv(B/'macro_summary.csv');project=pd.read_csv(B/'per_project_metrics.csv')
results=pd.read_csv(B/'outer_metrics.csv');sensitivity=pd.read_csv(B/'metrics_including_matching_test_vectors.csv')
importance=pd.read_csv(B/'importance_summary.csv');inventory=pd.read_csv(B/'data_inventory.csv')
folds=pd.read_csv(B/'fold_inventory.csv');selection=json.loads((B/'final_selection.json').read_text())
holdout=pd.read_csv(H/'holdout_metrics.csv');holdout_audit=json.loads((H/'audit.json').read_text())
seed_sd=pd.read_csv(B/'seed_standard_deviations.csv')
join=sensitivity.merge(results,on=['seed','test_project','condition','threshold_policy'],suffixes=('_all','_purged'))
scores=['accuracy','precision','recall','f1','roc_auc','average_precision']
for score in scores:join[score+'_difference']=join[score+'_all']-join[score+'_purged']
sens_summary=join.groupby(['condition','threshold_policy'])[[s+'_difference' for s in scores]].mean().reset_index()
sens_summary.to_csv(B/'overlap_sensitivity_summary.csv',index=False)
seed_summary=seed_sd.groupby(['condition','threshold_policy'])[[s for s in scores]].agg(['mean','max'])
seed_summary.columns=['_'.join(c) for c in seed_summary.columns]
seed_summary.reset_index().to_csv(B/'seed_variability_summary.csv',index=False)
correlations=pd.read_csv(B/'exploratory_correlations.csv')
correlations['absolute']=correlations.spearman.abs();correlations['strong']=correlations.absolute>=.8
corr=correlations.groupby(['feature_a','feature_b']).agg(median_absolute_rho=('absolute','median'),
                  minimum_absolute_rho=('absolute','min'),maximum_absolute_rho=('absolute','max'),
                  projects_ge_080=('strong','sum')).reset_index().sort_values('median_absolute_rho',ascending=False)
corr.to_csv(B/'correlation_pair_summary.csv',index=False)
distributions=pd.read_csv(B/'exploratory_distributions.csv')
descriptive=inventory[['project','clean_rows','positive_rate']].copy()
for feature in ['loc','rfc','cbo']:
 vals=distributions.loc[distributions.feature==feature,['project','median','q1','q3']].rename(columns={s:f'{feature}_{s}' for s in ['median','q1','q3']})
 descriptive=descriptive.merge(vals,on='project')
descriptive.to_csv(B/'exploratory_project_summary.csv',index=False)
threshold=project.loc[project.threshold_policy=='validation_F1'].merge(project.loc[project.threshold_policy=='fixed_0.5'],on=['condition','test_project'],suffixes=('_tuned','_fixed'))
for score in scores:threshold[score+'_change']=threshold[score+'_tuned']-threshold[score+'_fixed']
threshold.to_csv(B/'threshold_project_changes.csv',index=False)
threshold_summary=threshold.groupby('condition')[[s+'_change' for s in scores]].mean().reset_index()
threshold_summary.to_csv(B/'threshold_change_summary.csv',index=False)
selected=selection['condition']
selected_fixed=macro.loc[(macro.condition==selected)&(macro.threshold_policy=='fixed_0.5')].iloc[0].to_dict()
selected_tuned=macro.loc[(macro.condition==selected)&(macro.threshold_policy=='validation_F1')].iloc[0].to_dict()
external=holdout.loc[(holdout.condition==selected)&(holdout.population=='purged')]
external_summary=external.groupby('threshold_policy')[scores+['fn','tp','fp','tn']].mean().reset_index()
external_summary.to_csv(H/'mean_metrics_over_seeds.csv',index=False)
summary={
 'completion':json.loads((B/'completed.json').read_text()),'selection':selection,
 'inventory':inventory.to_dict('records'),'fold_inventory':folds.loc[folds.seed==42].to_dict('records'),
 'development_original_records':int(inventory.source_rows.sum()),
 'development_clean_records':int(inventory.clean_rows.sum()),
 'outer_test_records_per_seed':int(folds.loc[folds.seed==42].test_after.sum()),
 'outer_exclusions_per_seed':int((folds.loc[folds.seed==42].test_before-folds.loc[folds.seed==42].test_after).sum()),
 'selected_fixed_macro':selected_fixed,'selected_tuned_macro':selected_tuned,
 'macro_metrics':macro.to_dict('records'),'per_project_metrics':project.to_dict('records'),
 'threshold_changes':threshold_summary.to_dict('records'),
 'importance_top':importance.loc[importance.kind=='feature'].head(8).to_dict('records'),
 'group_importance':importance.loc[importance.kind=='group'].to_dict('records'),
 'importance_contexts':int(importance.loc[importance.kind=='feature','contexts'].iloc[0]),
 'correlations':corr.head(6).to_dict('records'),'descriptive':descriptive.to_dict('records'),
 'seed_variability':seed_summary.reset_index().to_dict('records'),
 'overlap_sensitivity':sens_summary.to_dict('records'),'holdout_audit':holdout_audit,
 'holdout_summary':external_summary.to_dict('records'),
 'holdout_baselines':holdout.loc[holdout.condition.isin(['Dummy','AlwaysPositive'])].to_dict('records'),
 'holdout_seed42':external.loc[external.seed==42].to_dict('records')}
def sanitize(value):
 if isinstance(value,dict):return {key:sanitize(item) for key,item in value.items()}
 if isinstance(value,list):return [sanitize(item) for item in value]
 if isinstance(value,float) and not np.isfinite(value):return None
 return value
(B/'report_summary.json').write_text(json.dumps(sanitize(summary),indent=2,allow_nan=False),encoding='utf-8')
print('Frozen external model:',selected,'parameter',selection['parameter'],'threshold',selection['secondary_threshold'])
print(macro.loc[macro.threshold_policy=='fixed_0.5',['condition','recall_mean','f1_mean','roc_auc_mean','average_precision_mean']].to_string(index=False))
print(external_summary.to_string(index=False))
