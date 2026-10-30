"""Score saved incumbent outputs on precisely our task IDs with our independent metrics."""
import json,argparse
from pathlib import Path
from .experiment import tasks,measures,WIDE
p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();root=Path('results')/a.tag
records=[json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()];srcs={n:s for n,s,t,ref,f in tasks('all')};ids={r['name'] for r in records};result={}
for method in ('main','sprint'):
    baseline=json.loads((WIDE/'bench'/method/'results.json').read_text());ms=[]
    for name in sorted(ids):
        if name in baseline and baseline[name].get('out'):ms.append(measures(srcs[name],baseline[name]['out']))
    result[method]={k:round(sum(x[k] for x in ms),4) for k in ('hard','hidden','small','text_overlap')};result[method]['n']=len(ms)
(root/'baselines.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
