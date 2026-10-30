"""Independent valid-schema fuzz corpus and metamorphic checks, no incumbent imports."""
import random,json,time,os,argparse,math,resource
from pathlib import Path
from engine.resize import resize
from engine.objective import violations
os.environ['ORIGIN_PIXELS']='0'
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=100);p.add_argument('--tag',default='stress');a=p.parse_args();root=Path('results')/a.tag;root.mkdir(parents=True,exist_ok=True);path=root/'records.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [];seen={r['i'] for r in rows};rng=random.Random(20261030)
for z in range(a.n):
    W,H=rng.choice([(800,600),(1000,1000),(400,900)]);els=[dict(id='b',type='shape',x=0,y=0,width=W,height=H,shape_type='rectangle',fill_color='rgba(250,250,250,1)',z_index=0)]
    for k in range(rng.randint(1,12)):
        x,y=rng.random()*W*.8,rng.random()*H*.8;w,h=rng.uniform(.1,.4)*W,rng.uniform(.04,.2)*H
        if rng.random()<.65:
            fs=rng.randint(10,50);e=dict(id=str(k),type='text',x=x,y=y,width=w,height=h,font_family='Solstice Sans',font_size=fs,content=' '.join(rng.choices(['Summer','light','extraordinary','garden','collection'],k=rng.randint(1,12))),color='rgba(30,20,30,1)',z_index=k+1,text_align=rng.choice(['left','center','right']))
        else:e=dict(id=str(k),type='image',x=x,y=y,width=w,height=h,asset_id='solstice_bottle.png',fit='contain',z_index=k+1)
        els.append(e)
    src=dict(canvas=dict(width=W,height=H),elements=els);tw,th=rng.choice([(728,90),(300,250),(1080,1920),(1600,900),(200,600)]);target=dict(width=tw,height=th)
    if z in seen:continue
    start=time.monotonic()
    try:
        out=resize(src,target);errors=violations(src,out);crash=None
    except Exception as e:errors=[];crash=repr(e)
    record=dict(i=z,errors=errors,crash=crash,seconds=round(time.monotonic()-start,3))
    if crash or errors:record.update(source=src,target=target)
    rows.append(record)
    with path.open('a') as f:f.write(json.dumps(record)+'\n')
    print(z,record['seconds'],crash,errors,flush=True)
summary=dict(n=len(rows),crashes=sum(bool(r['crash']) for r in rows),failures=sum(bool(r['errors']) for r in rows),max_seconds=max(r['seconds'] for r in rows),peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),cpu_seconds=round(time.process_time(),3));(root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
