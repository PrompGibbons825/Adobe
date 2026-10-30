"""Run with the incumbent repo as cwd, keeping its checker outside our runtime.
This adapter reads only our saved output records and public development sources.
"""
import sys,json,os
from pathlib import Path
sys.path.insert(0,'/home/vlatko/code/adobe-hackathon-2026/chameleon')
from tools.rules import hard_rule_errors,promise_errors,ghost_text
root=Path('/data/agents/work/codex-origin-20261030');wide=Path('/data/agents/work/crello-wide');tag=sys.argv[1]
sources={}
for p in (wide/'cases').glob('*.json'):
    d=json.loads(p.read_text());sources[d['family']]=d['source']
records=[json.loads(x) for x in (root/'results'/tag/'records.jsonl').read_text().splitlines()]
counts={'tasks':len(records),'hard':0,'promise':0,'ghost':0};rows=[]
for r in records:
    src=sources[r['family']];out=r['output'];h=hard_rule_errors(src,out,out['canvas']);p=promise_errors(src,out,out['canvas'])
    g=[]
    # We never hide text, so no opacity-zero copy needs a ghost render.
    if any(e['type']=='text' and e.get('opacity',1)==0 for e in out['elements']):g=ghost_text(out,wide/'cases/assets')
    counts['hard']+=len(h);counts['promise']+=len(p);counts['ghost']+=len(g)
    if h or p or g:rows.append({'name':r['name'],'hard':h,'promise':p,'ghost':g})
(root/'results'/tag/'external-check.json').write_text(json.dumps({'summary':counts,'failures':rows},indent=2));print(counts)
