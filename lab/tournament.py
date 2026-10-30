import os,time,json,argparse,statistics,resource
from pathlib import Path
from engine.resize import diagnose_all
from .experiment import tasks,measures,WIDE

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--limit',type=int,default=24);p.add_argument('--tag',default='round1');p.add_argument('--no-pixels',action='store_true');a=p.parse_args()
    os.environ['ORIGIN_PRIOR']='1'
    os.environ['ORIGIN_ASSETS']=str(WIDE/'cases/assets');dest=Path('results')/a.tag;dest.mkdir(parents=True,exist_ok=True);path=dest/'records.jsonl';records=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [];seen={r['name'] for r in records}
    for k,(name,src,t,ref,family) in enumerate(tasks(a.split,a.limit)):
        if name in seen:continue
        start=time.monotonic();winners,count=diagnose_all(src,t,not a.no_pixels)
        record=dict(name=name,family=family,seconds=round(time.monotonic()-start,3),generated=count,methods={m:dict(winner=p['name'],metrics=measures(src,p['output']),output=p['output']) for m,p in winners.items()})
        with path.open('a') as f:f.write(json.dumps(record)+'\n')
        records.append(record);print(k+1,name,round(record['seconds'],2),{m:x['metrics'] for m,x in record['methods'].items()},flush=True)
    summary={}
    for m in ('elastic','hierarchy','search','retrieval','portfolio'):
        ms=[r['methods'][m]['metrics'] for r in records]
        summary[m]=dict(n=len(ms),hard=sum(x['hard'] for x in ms),hidden=sum(x['hidden'] for x in ms),small=sum(x['small'] for x in ms),text_overlap=round(sum(x['text_overlap'] for x in ms),3),median_font=round(statistics.median(x['median_font'] for x in ms),1))
        out=Path('results')/(a.tag+'-'+m);out.mkdir(exist_ok=True)
        (out/'records.jsonl').write_text('\n'.join(json.dumps(dict(name=r['name'],family=r['family'],seconds=r['seconds'],**r['methods'][m])) for r in records)+'\n')
    summary['metadata']=dict(split=a.split,families=len({r['family'] for r in records}),n=len(records),seconds_median=statistics.median(r['seconds'] for r in records),peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),cpu_seconds=round(time.process_time(),3))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
