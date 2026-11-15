"""Post hoc project-level hypothesis checks using archived, executed predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'Assignment3_Practical/outputs_benchmark/per_project_metrics.csv'

def main():
    data=pd.read_csv(SOURCE)
    fixed=data.loc[data.threshold_policy=='fixed_0.5']
    baseline=fixed.loc[fixed.condition=='Dummy'].set_index('test_project').average_precision
    rows=[];differences=[]
    for condition in ['LR_CK6','LR_all20','RF_CK6','RF_all20','XGB_CK6','XGB_all20']:
        scores=fixed.loc[fixed.condition==condition].set_index('test_project').sort_index()
        delta=scores.average_precision-baseline.reindex(scores.index)
        test=wilcoxon(delta,alternative='two-sided',method='exact')
        rows.append({'condition':condition,'projects':len(delta),
                     'mean_ap_gain':float(delta.mean()),'median_ap_gain':float(delta.median()),
                     'min_ap_gain':float(delta.min()),'max_ap_gain':float(delta.max()),
                     'wilcoxon_statistic':float(test.statistic),'p_two_sided':float(test.pvalue)})
        for project,value in delta.items():
            differences.append({'condition':condition,'project':project,'ap_difference_from_dummy':float(value)})
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_two_sided'])
    running=0.0
    for rank,index in enumerate(order):
        running=max(running,min(1.0,(len(rows)-rank)*rows[index]['p_two_sided']))
        rows[index]['p_holm']=float(running)
        rows[index]['reject_at_005']=bool(running<.05)
    pd.DataFrame(rows).to_csv(OUT/'exploratory_hypothesis_tests.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'project_ap_differences.csv',index=False)
    record={'executed_utc':datetime.now(timezone.utc).isoformat(),'status':'executed',
            'analysis_stage':'Final-paper extension after inspection of Assignment 4 results; exploratory, not preregistered.',
            'hypothesis':'For each learned condition, H0: paired project AP differences from the prior baseline are centred on zero; H1: they are not.',
            'method':'Two-sided exact Wilcoxon signed-rank; five seed-averaged project pairs; Holm correction over six tests; alpha 0.05.',
            'limitations':'Convenience sample of five projects with overlapping training folds; symmetry and exchangeability assumptions and model-selection history limit inference. Seeds are not independent projects.',
            'input_sha256_lf_normalized':hashlib.sha256(SOURCE.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            'input_hash_normalization':'CRLF converted to LF so the audit survives Git text normalization.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows}
    (OUT/'statistical_analysis.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
