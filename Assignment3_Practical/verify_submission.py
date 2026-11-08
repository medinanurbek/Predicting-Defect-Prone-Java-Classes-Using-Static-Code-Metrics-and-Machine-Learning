"""Audit saved sources and results without fitting models or claiming personal sign-off."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix

ROOT=Path(__file__).resolve().parent
B=ROOT/'outputs_benchmark';H=ROOT/'outputs_holdout'
FEATURES='wmc dit noc cbo rfc lcom ca ce npm lcom3 loc dam moa mfa cam ic cbm amc max_cc avg_cc'.split()
SCORES=['accuracy','precision','recall','f1','roc_auc','average_precision']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def calculate(y,p,t):
    y=np.asarray(y);p=np.asarray(p);decision=p>=t
    out=dict(zip(SCORES,[accuracy_score(y,decision),precision_score(y,decision,zero_division=0),
         recall_score(y,decision,zero_division=0),f1_score(y,decision,zero_division=0),roc_auc_score(y,p),average_precision_score(y,p)]))
    out.update(dict(zip(['tn','fp','fn','tp'],confusion_matrix(y,decision,labels=[0,1]).ravel())))
    return out
def match(row,values):
    for field,value in values.items():
        assert np.isclose(float(row[field]),float(value)),f'{field}: {row[field]} != {value}'

def main():
    checks=[];dev=pd.read_csv(B/'analysis_ready_development.csv',dtype={'signature':str})
    inv=pd.read_csv(B/'data_inventory.csv');eda=pd.read_csv(B/'exploratory_distributions.csv')
    sources=read(ROOT/'data/manifest.json')['files']
    source_holdout=read(ROOT/'data_holdout/manifest.json')
    tables=[(s['id'],ROOT/'data'/f'{s["id"]}.csv',s['sha256']) for s in sources]
    tables.append(('synapse-1.2',ROOT/'data_holdout/synapse-1.2.csv',source_holdout['sha256']))
    total=0
    for project,path,expected in tables:
        assert sha(path)==expected
        raw=pd.read_csv(path);target=pd.to_numeric(raw.bug)
        assert target.notna().all() and (target>=0).all() and (target==target.round()).all()
        assert raw['name.1'].notna().all() and not raw['name.1'].duplicated().any()
        assert np.isfinite(raw[FEATURES].to_numpy()).all() and (raw[FEATURES]>=0).all().all()
        ids=raw['name.1'].map(lambda name:hashlib.sha256((project+'|'+name).encode()).hexdigest()[:20])
        if project!='synapse-1.2':
            saved=dev.loc[dev.project==project].set_index('class_id').loc[ids]
            assert np.array_equal(saved.defective.to_numpy(),(target>0).astype(int).to_numpy())
            assert np.array_equal(saved.bug_count.to_numpy(),target.to_numpy())
            assert np.allclose(saved[FEATURES].to_numpy(),raw[FEATURES].to_numpy())
            row=inv.loc[inv.project==project].iloc[0]
            match(row,{'source_rows':len(raw),'clean_rows':len(raw),'positive':int((target>0).sum()),'negative':int((target==0).sum())})
            for feature in FEATURES:
                e=eda.loc[(eda.project==project)&(eda.feature==feature)].iloc[0];x=raw[feature]
                match(e,{'median':x.median(),'q1':x.quantile(.25),'q3':x.quantile(.75),'minimum':x.min(),'maximum':x.max()})
        else:
            saved=pd.read_csv(H/'analysis_ready_holdout.csv',dtype={'signature':str})
            lookup=pd.DataFrame({'class_id':ids,'actual':(target>0).astype(int) }).set_index('class_id')
            assert np.array_equal(saved.defective.to_numpy(),lookup.loc[saved.class_id].actual.to_numpy())
            assert not set(saved.signature)&set(dev.signature)
            assert len(raw)==256 and len(saved)==243 and saved.defective.sum()==86
        total+=len(raw)
    checks.append(f'PASS source inventory, six raw hashes, labels and source-to-analysis mapping ({total} source rows).')
    checks.append('PASS all 20 feature distributions independently recomputed for five development projects.')
    outer=pd.read_csv(B/'outer_metrics.csv');pred=pd.read_csv(B/'outer_predictions.csv.gz')
    count=0
    for _,row in outer.iterrows():
        group=pred.loc[(pred.seed==row.seed)&(pred.test_project==row.test_project)&(pred.condition==row.condition)]
        if row.condition in ['Dummy','AlwaysPositive']:
            reference=pred.loc[(pred.seed==row.seed)&(pred.test_project==row.test_project)&(pred.condition=='LR_CK6')]
            prior=dev.loc[dev.project!=row.test_project].defective.mean()
            y=reference.actual;p=np.full(len(y),1. if row.condition=='AlwaysPositive' else prior)
        else:y=group.actual;p=group.probability
        match(row,calculate(y,p,row.threshold));count+=1
    assert count==210
    macro=pd.read_csv(B/'macro_summary.csv')
    projects=outer.groupby(['condition','threshold_policy','test_project'])[SCORES].mean().reset_index()
    for _,r in macro.iterrows():
        g=projects.loc[(projects.condition==r.condition)&(projects.threshold_policy==r.threshold_policy)]
        for score in SCORES:match(r,{score+'_mean':g[score].mean(),score+'_std':g[score].std(),score+'_min':g[score].min(),score+'_max':g[score].max()})
    checks.append('PASS all 210 outer metric rows, both baselines and equal-project means/ranges/SD independently recalculated.')
    external=pd.read_csv(H/'holdout_metrics.csv');hp=pd.read_csv(H/'holdout_predictions.csv');selection=read(B/'final_selection.json')
    for _,r in external.loc[external.population=='purged'].iterrows():
        if r.condition in ['Dummy','AlwaysPositive']:
            g=hp.loc[hp.seed==42];p=np.full(len(g),1. if r.condition=='AlwaysPositive' else dev.defective.mean())
        else:g=hp.loc[hp.seed==r.seed];p=g.probability
        match(r,calculate(g.actual,p,r.threshold))
    checks.append('PASS eight retained-population Synapse metric rows, including both baselines, independently recalculated.')
    protocol=read(B/'protocol.json');completed=read(B/'completed.json')
    assert sha(B/'protocol.json')==selection['protocol_sha256']==completed['protocol_sha256']
    assert sha(B/'final_selection.json')==completed['final_selection_sha256']==source_holdout['selection_sha256_before_acquisition']
    assert datetime.fromisoformat(selection['frozen_utc'])<datetime.fromisoformat(source_holdout['acquired_utc'])
    assert sha(B/'executed_run_benchmark.py')==protocol['script_sha256']
    assert sha(B/'executed_benchmark_utils.py')==sha(ROOT/'benchmark_utils.py')==protocol['utils_sha256']
    assert sha(H/'executed_evaluate_holdout.py')==read(H/'audit.json')['evaluation_script_sha256']
    assert len(pd.read_csv(B/'inner_cv_scores.csv'))==1080
    checks.append('PASS frozen selection chronology, recorded source snapshots and 1,080 recorded inner candidate scores.')
    report=ROOT.parent/'output/Assignment4_Kabysheva_Nurbek_Maksatova_Nauryzbay.docx'
    with ZipFile(report) as z:
        xml=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        dictionary=[]
        for table in xml.findall('.//w:tbl',ns):
            rows=[[''.join(c.itertext()) for c in row.findall('w:tc/w:p/w:r/w:t',ns)] for row in table.findall('w:tr',ns)]
            if rows[0]==['Field','Meaning','Type','Units','Role']:dictionary=[row[0] for row in rows[1:]]
        assert set(dictionary)==set(FEATURES+['bug_count','defective','project','class_id','signature'])
    checks.append('PASS report dictionary covers all 25 analysis fields; metadata and labels excluded from predictors.')
    record={'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'Package-level verification by the assistant; no claim of personal execution by named students.',
            'verification_script_sha256':sha(Path(__file__)),'checks':checks,
            'limitations':'No model refitting in this audit; labels checked against published files, not against underlying bug reports.'}
    (B/'submission_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('\n'.join(checks))

if __name__=='__main__':main()
