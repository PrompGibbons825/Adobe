"""Cross-family relationship transfer. Only coordinates and type ranks are stored.
This generator supplies hypotheses; the common objective can reject all of them.
"""
import json,math
from pathlib import Path
from functools import lru_cache
from .geometry import Rect
@lru_cache(maxsize=1)
def library():
    path=Path(__file__).parent/'data/designer_prior.json'
    return json.loads(path.read_text()) if path.exists() else []

def retrieve(s,W,H):
    if len(s.islands)>14:return
    def rank(k):
        q=s.islands[k]
        fs=max([float(s.els[i].get('font_size',0)) for i in q.ids]+[0])
        return ('text' if q.text else 'art',-fs,-q.rect.area)
    ids=sorted(range(len(s.islands)),key=rank);kinds=[rank(k)[0] for k in ids]
    ar0=s.W/s.H;ar=W/H;scored=[]
    for row in library():
        items=row['items']
        if len(items)!=len(ids) or [it['kind'] for it in items]!=kinds:continue
        error=2*abs(math.log(row['source_ar']/ar0))+3*abs(math.log(row['target_ar']/ar))
        for k,it in zip(ids,items):
            r=s.islands[k].rect;v=[r.x/s.W,r.y/s.H,r.w/s.W,r.h/s.H]
            error+=sum(abs(a-b) for a,b in zip(v,it['src']))/max(1,len(ids))
        scored.append((error,row))
    for error,row in sorted(scored,key=lambda p:p[0])[:4]:
        if error>3:continue
        cells={}
        for k,it in zip(ids,row['items']):
            x,y,w,h=it['dst'];cells[k]=Rect(max(0,x*W),max(0,y*H),max(1,min(1-x,w)*W),max(1,min(1-y,h)*H))
        yield 'retrieval',cells
