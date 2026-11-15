"""Read-only consistency audit for the course final-paper package."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from itertools import product
import csv,json,re,hashlib

ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def text(element):return ''.join(x.text or '' for x in element.findall('.//w:t',NS))
def main():
    refs=read(ROOT/'Final_Paper/references/verified_references.json')
    reference_count=len(refs)
    report=ROOT/'output/Final_Paper_Kabysheva_Nurbek_Maksatova_Nauryzbay.docx'
    with ZipFile(report) as z:
        body=ET.fromstring(z.read('word/document.xml'))
        paragraphs=[text(p) for p in body.findall('w:body/w:p',NS)]
        visible=' '.join(text(p) for p in body.findall('.//w:p',NS))
        assert not re.search('[\u0400-\u04ff]',visible),'Non-English Cyrillic article text'
        assert 'Identify applicable sponsor' not in visible
        assert visible.count('Email:')==4
        assert len(body.findall('.//w:drawing',NS))==2
        tables=[]
        for table in body.findall('w:body/w:tbl',NS):
            tables.append([[text(cell) for cell in row.findall('w:tc',NS)] for row in table.findall('w:tr',NS)])
        ref_start=paragraphs.index('References')
        reference_lines=[p for p in paragraphs[ref_start+1:] if re.match(r'^\[\d+\]',p)]
        numbers=[int(re.match(r'^\[(\d+)\]',p).group(1)) for p in reference_lines]
        assert reference_count>=20 and numbers==list(range(1,reference_count+1))
        used=[]
        for number in map(int,re.findall(r'\[(\d+)\]',' '.join(paragraphs[:ref_start]))):
            if number not in used:used.append(number)
        assert used==list(range(1,reference_count+1)),'First-citation order or coverage'
        assert sum('TABLE '+n+'.' in p for p in paragraphs for n in ['I','II','III','IV'])==4
        assert sum(p.startswith('Fig. ') for p in paragraphs)==2
    print(f'PASS article structure: English text, four email fields, {reference_count} ordered/cited references, four tables and two figures.')
    index=read(ROOT/'Final_Paper/references/indexing_evidence.json')
    indexed=sum(r['counted_for_indexing_minimum'] for r in refs)
    assert index['reference_count']==reference_count and index['conservatively_counted_indexed_journal_references']==indexed
    assert index['fraction']>=.30
    assert all(r.get('url') and r.get('metadata_verification') for r in refs)
    assert abs(index['fraction']-indexed/reference_count)<1e-12
    recent=sum(2020<=r['year']<=2026 for r in refs)
    assert recent==index['references_2020_to_2026'] and recent>reference_count/2
    print(f'PASS bibliography: {reference_count} verified records, {recent} from 2020-2026, and {indexed} conservatively counted journal-indexing items.')
    project=rows(ROOT/'Assignment3_Practical/outputs_benchmark/per_project_metrics.csv')
    selected=[r for r in project if r['threshold_policy']=='fixed_0.5']
    baseline={r['test_project']:float(r['average_precision']) for r in selected if r['condition']=='Dummy'}
    testrows=rows(ROOT/'Final_Paper/analysis/exploratory_hypothesis_tests.csv')
    # Independent enumeration of the exact null signed-rank distribution for n=5.
    ranks=[1,2,3,4,5]
    distribution=[sum(rank for rank,sign in zip(ranks,signs) if sign) for signs in product([0,1],repeat=5)]
    exact=sum(value in [0,sum(ranks)] for value in distribution)/len(distribution)
    assert exact==.0625
    for row in testrows:
        own=[r for r in selected if r['condition']==row['condition']]
        diff=[float(r['average_precision'])-baseline[r['test_project']] for r in own]
        assert len(diff)==5 and all(x>0 for x in diff)
        assert abs(sum(diff)/5-float(row['mean_ap_gain']))<1e-12
        assert float(row['p_two_sided'])==exact
        assert float(row['p_holm'])==min(1,6*exact)==.375
    assert len(testrows)==6
    audit=read(ROOT/'Final_Paper/analysis/statistical_analysis.json')
    source=ROOT/'Assignment3_Practical/outputs_benchmark/per_project_metrics.csv'
    assert hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==audit['input_sha256_lf_normalized']
    print('PASS exploratory tests: five paired project means, exact p=0.0625 and Holm p=0.375; input hash verified.')
    model_table=next(t for t in tables if t[0][0]=='Model')
    macro={r['condition']:r for r in rows(ROOT/'Assignment3_Practical/outputs_benchmark/macro_summary.csv') if r['threshold_policy']=='fixed_0.5'}
    labels={'LR6':'LR_CK6','LR20':'LR_all20','RF6':'RF_CK6','RF20':'RF_all20','XGB6':'XGB_CK6','XGB20':'XGB_all20','Prior':'Dummy','All positive':'AlwaysPositive'}
    for r in model_table[1:]:
        original=macro[labels[r[0]]]
        for actual,field in zip(r[1:4],['average_precision_mean','roc_auc_mean','f1_mean']):assert actual==f'{float(original[field]):.3f}'
    synapse=rows(ROOT/'Assignment3_Practical/outputs_holdout/holdout_predictions.csv')
    one=[r for r in synapse if r['seed']=='42']
    if not one:one=[r for r in synapse if float(r['seed'])==42]
    assert len(one)==243 and sum(int(r['actual']) for r in one)==86
    for threshold,expected in [(.5,(132,25,48,38)),(.4,(91,66,25,61))]:
        counts=[0,0,0,0]
        for r in one:
            y=int(r['actual']);pred=float(r['probability'])>=threshold
            counts[(2 if y else 0)+(1 if pred else 0)]+=1
        assert tuple(counts)==expected
    print('PASS reported comparison scores and both Synapse confusion matrices match saved results.')
    print('Email completion, slide template/timing, individual understanding and Moodle plagiarism review require the team.')

if __name__=='__main__':main()
