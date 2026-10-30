"""Fixed family-disjoint comparison. Never reads sealed data or writes shared state."""
import os,json,math,time,argparse,hashlib,statistics
from pathlib import Path
from engine.resize import diagnose
from engine.objective import violations
from engine.typefit import painted as footprint
from engine.geometry import box,finite
from renderer.converter.convert import render_design_to_png

WIDE=Path('/data/agents/work/crello-wide')

def tasks(split='dev',limit=0):
    selected=set(json.loads((WIDE/'tournament_set.json').read_text())['tasks'])
    all=[]
    for path in sorted((WIDE/'cases').glob('*.json')):
        f=json.loads(path.read_text());bucket=int(hashlib.sha256(f['family'].encode()).hexdigest()[:8],16)%3
        if split=='dev' and bucket==0:continue
        if split=='holdout' and bucket!=0:continue
        for k,ref in enumerate(f['targets']):
            name=f"{f['family']}_{k}"
            if name in selected:all.append((name,f['source'],ref['canvas'],ref,f['family']))
    return all[:limit or None]

def measures(src,out):
    W,H=out['canvas']['width'],out['canvas']['height'];floor=max(10,10*max(W,H)/1024);texts=[e for e in out['elements'] if e['type']=='text' and str(e.get('content','')).strip()]
    visible=[e for e in texts if finite(e.get('opacity'),1)>0];overlap=0
    so={e['id']:e for e in src['elements']}
    for z,e in enumerate(visible):
        for f in visible[:z]:
            a,b=footprint(e),footprint(f)
            sa,sb=footprint(so[e['id']]),footprint(so[f['id']])
            if sa.inter(sb)<.05*min(sa.area,sb.area):
                overlap+=a.inter(b)/max(1,min(a.area,b.area))
    return dict(hard=len([x for x in violations(src,out) if x!='changed:opacity' and not x.startswith('glyph:')]),hidden=sum(finite(e.get('opacity'),1)==0 for e in out['elements']),small=sum(e.get('font_size',0)<floor for e in visible),text_overlap=round(overlap,4),median_font=statistics.median([e['font_size'] for e in visible] or [0]))

def run(args):
    os.environ['ORIGIN_ASSETS']=str(WIDE/'cases/assets');dest=Path('results')/args.tag;dest.mkdir(parents=True,exist_ok=True);records=[]
    existing={}
    if (dest/'records.jsonl').exists():existing={r['name']:r for r in map(json.loads,(dest/'records.jsonl').read_text().splitlines())}
    for n,(name,src,t,ref,family) in enumerate(tasks(args.split,args.limit)):
        if name in existing:records.append(existing[name]);continue
        start=time.monotonic();best,info=diagnose(src,t,args.method,pixels=not args.no_pixels);out=best['output'];sec=time.monotonic()-start
        record=dict(name=name,family=family,seconds=round(sec,3),winner=best['name'],generated=info['generated'],metrics=measures(src,out),output=out)
        with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        records.append(record)
        if args.render and n<12:
            render_design_to_png(out,WIDE/'cases/assets',dest/(name+'.png'),fonts_dir='brand_kit/fonts',max_size=1000)
            render_design_to_png(src,WIDE/'cases/assets',dest/(name+'.src.png'),fonts_dir='brand_kit/fonts',max_size=1000)
        print(args.tag,n+1,name,record['winner'],record['metrics'],round(sec,2),flush=True)
    summary=dict(tag=args.tag,method=args.method,split=args.split,n=len(records),families=len({r['family'] for r in records}),hard=sum(r['metrics']['hard'] for r in records),hard_tasks=sum(r['metrics']['hard']>0 for r in records),hidden=sum(r['metrics']['hidden'] for r in records),small=sum(r['metrics']['small'] for r in records),text_overlap=sum(r['metrics']['text_overlap'] for r in records),seconds_median=statistics.median([r['seconds'] for r in records]) if records else None)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--method',default='portfolio',choices=['elastic','hierarchy','search','portfolio']);p.add_argument('--split',default='dev',choices=['dev','holdout','all']);p.add_argument('--limit',type=int,default=0);p.add_argument('--no-pixels',action='store_true');p.add_argument('--render',action='store_true');run(p.parse_args())
