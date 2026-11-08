"""Acquire and evaluate the additional Synapse project after selection is locked."""
import argparse
import json
import sys
import zipfile
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from benchmark_utils import ROOT,sha,load_development,load_table,columns,build,metric,purge,verify_pipeline
from fetch_data import RemoteZip,URL

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--benchmark-output',default='outputs_benchmark');parser.add_argument('--output',default='outputs_holdout');args=parser.parse_args()
 benchmark=ROOT/args.benchmark_output;out=ROOT/args.output
 if out.exists() and any(out.iterdir()):raise RuntimeError('Use an empty --output directory to preserve existing holdout evidence.')
 completed=json.loads((benchmark/'completed.json').read_text())
 selection_path=benchmark/'final_selection.json';selection=json.loads(selection_path.read_text())
 if sha(selection_path)!=completed['final_selection_sha256']:raise ValueError('Final selection changed after benchmark completion')
 protocol=json.loads((benchmark/'protocol.json').read_text())
 if sha(benchmark/'protocol.json')!=selection['protocol_sha256']:raise ValueError('Protocol changed')
 if sha(ROOT/'benchmark_utils.py')!=protocol['utils_sha256']:raise ValueError('Model implementation changed after selection')
 dev,inventory=load_development()
 for row in inventory.to_dict('records'):
  if protocol['source_sha256'][row['project']]!=row['sha256']:raise ValueError('Development data changed')
 out.mkdir(exist_ok=True,parents=True)
 # Save an exact selection copy before any new archive member is acquired.
 (out/'selection_before_acquisition.json').write_bytes(selection_path.read_bytes())
 data_dir=ROOT/'data_holdout';data_dir.mkdir(exist_ok=True)
 source=data_dir/'synapse-1.2.csv';manifest_path=data_dir/'manifest.json'
 if source.exists():
  manifest=json.loads(manifest_path.read_text())
  if manifest['selection_sha256_before_acquisition']!=sha(selection_path):raise ValueError('This holdout was acquired under another selection')
  if sha(source)!=manifest['sha256']:raise ValueError('Holdout checksum mismatch')
 else:
  if datetime.fromisoformat(selection['frozen_utc'])>=datetime.now(timezone.utc):raise ValueError('Invalid selection timestamp')
  member=protocol['future_holdout_member']
  with zipfile.ZipFile(RemoteZip()) as archive:content=archive.read(member)
  source.write_bytes(content)
  manifest={'project':'synapse-1.2','record_doi':'10.5281/zenodo.3693686','archive_url':URL,'member':member,
            'acquired_utc':datetime.now(timezone.utc).isoformat(),'bytes':len(content),'sha256':sha(source),
            'archive_crc_checked':True,'selection_sha256_before_acquisition':sha(selection_path),
            'selection_frozen_utc':selection['frozen_utc']}
  manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 holdout,source_audit=load_table(source,'synapse-1.2');retained=purge(holdout,dev)
 retained.to_csv(out/'analysis_ready_holdout.csv',index=False)
 if set(dev.project)&set(retained.project):raise AssertionError('Project overlap')
 if set(dev.signature)&set(retained.signature):raise AssertionError('Exact-vector overlap')
 if retained.defective.nunique()!=2:raise ValueError('Holdout needs both classes')
 condition=selection['condition'];param=selection['parameter'];cols=columns(condition)
 rows=[];predictions=[]
 for seed in selection['seeds']:
  model=build(condition,param,seed,dev.defective).fit(dev[cols],dev.defective);verify_pipeline(model,dev,cols)
  prob=model.predict_proba(retained[cols])[:,1]
  prob_all=model.predict_proba(holdout[cols])[:,1]
  for policy,threshold in [('fixed_0.5',.5),('validation_F1',selection['secondary_threshold'])]:
   rows.append({'seed':seed,'condition':condition,'threshold_policy':policy,'population':'purged',**metric(retained.defective,prob,threshold)})
   rows.append({'seed':seed,'condition':condition,'threshold_policy':policy,'population':'all_records',**metric(holdout.defective,prob_all,threshold)})
  predictions.append(pd.DataFrame({'seed':seed,'condition':condition,'class_id':retained.class_id.to_numpy(),
                     'actual':retained.defective.to_numpy(),'probability':prob,'fixed_prediction':(prob>=.5).astype(int),
                     'validation_threshold':selection['secondary_threshold'],
                     'validation_prediction':(prob>=selection['secondary_threshold']).astype(int)}))
 for condition_reference,probability in [('Dummy',float(dev.defective.mean())),('AlwaysPositive',1.0)]:
  rows.append({'seed':None,'condition':condition_reference,'threshold_policy':'fixed_0.5','population':'purged',
               **metric(retained.defective,np.full(len(retained),probability),.5)})
 results=pd.DataFrame(rows);results.to_csv(out/'holdout_metrics.csv',index=False)
 predictions=pd.concat(predictions,ignore_index=True);predictions.to_csv(out/'holdout_predictions.csv',index=False)
 for seed,g in predictions.groupby('seed'):
  for policy,threshold in [('fixed_0.5',.5),('validation_F1',selection['secondary_threshold'])]:
   saved=results.loc[(results.seed==seed)&(results.threshold_policy==policy)&(results.population=='purged')].iloc[0]
   calc=metric(g.actual,g.probability,threshold)
   if not all(np.isclose(saved[k],calc[k]) for k in ['accuracy','precision','recall','f1','roc_auc','average_precision']):raise AssertionError('Metric recomputation mismatch')
 audit={'source':source_audit,'test_before_overlap_removal':len(holdout),'test_after_overlap_removal':len(retained),
        'excluded_matching_vectors':len(holdout)-len(retained),'development_training_rows':len(dev),
        'selection':selection,'acquisition':manifest,'evaluation_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_script_sha256':sha(__file__),
        'limitation':'Additional untouched project from the same historical archive; not an independent modern dataset family.'}
 (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 checks=['PASS selection and protocol hashes verified before holdout acquisition.',
         'PASS original development checksums unchanged.',
         'PASS raw holdout archive CRC and SHA256 recorded.',
         'PASS project and exact-vector separation.',
         'PASS no holdout data used to select model, parameters or threshold.',
         'PASS training-only preprocessing for every seed.',
         'PASS metrics recomputed from saved prediction records.']
 (out/'checks.txt').write_text('\n'.join(checks)+'\n',encoding='utf-8')
 print('Additional holdout:',len(holdout),'original;',len(retained),'retained;',int(retained.defective.sum()),'positive',flush=True)
 print('Frozen model:',condition,'parameter',param,'secondary threshold',selection['secondary_threshold'],flush=True)
 print(results.loc[results.population=='purged',['seed','condition','threshold_policy','precision','recall','f1','roc_auc','average_precision','fn','tp']].to_string(index=False),flush=True)

if __name__=='__main__':main()
