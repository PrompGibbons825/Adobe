"""Paired development/holdout reports, with explicit retention/readability tradeoffs."""
import json,statistics,math,argparse,collections
from pathlib import Path
from .experiment import tasks,measures,WIDE
from engine.geometry import box,finite

def artwork(src,out):
    sw,sh=src['canvas']['width'],src['canvas']['height'];ids={e['id'] for e in src['elements'] if e['type']=='image' and box(e).area<.8*sw*sh}
    return sum(box(e).area*finite(e.get('opacity'),1) for e in out['elements'] if e['id'] in ids)

p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();root=Path('results')/a.tag;records=[json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()];srcs={n:s for n,s,t,r,f in tasks('all')};result={};paired=[]
for method in ('main','sprint'):
    saved=json.loads((WIDE/'bench'/method/'results.json').read_text());ms=[];ratios=[];byname=[]
    for r in records:
        name=r['name']
        if name not in saved or not saved[name].get('out'):continue
        src=srcs[name];base=saved[name]['out'];ours=r['methods']['portfolio']['output'];m=measures(src,base);o=measures(src,ours);ms.append(m)
        before,after=artwork(src,base),artwork(src,ours)
        if before>0 and after>0:ratios.append(after/before);byname.append(dict(name=name,ratio=round(after/before,3)))
        paired.append(dict(name=name,baseline=method,base=m,ours=o))
    result[method]=dict(n=len(ms),hidden=sum(x['hidden'] for x in ms),small=sum(x['small'] for x in ms),new_text_overlap=round(sum(x['text_overlap'] for x in ms),4),artwork_area_median_ratio=round(statistics.median(ratios),3) if ratios else None,artwork_10x_tasks=sum(x>=10 for x in ratios),artwork_100x_tasks=sum(x>=100 for x in ratios),largest_artwork_gains=sorted(byname,key=lambda r:-r['ratio'])[:5])
ms=[r['methods']['portfolio']['metrics'] for r in records];result['origin']=dict(n=len(ms),hidden=sum(x['hidden'] for x in ms),small=sum(x['small'] for x in ms),new_text_overlap=round(sum(x['text_overlap'] for x in ms),4),selected=collections.Counter(r['methods']['portfolio']['winner'].split('/')[0] for r in records))
(root/'comparison.json').write_text(json.dumps(dict(summary=result,paired=paired),indent=2));print(json.dumps(result,indent=2))
