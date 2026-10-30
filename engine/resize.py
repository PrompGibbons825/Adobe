"""Origin: relationship-preserving design retargeting built independently.

Contract: resize(source,target_canvas)->design. No online calls, no asset changes,
no omission of elements. All candidates are realized and finalized before scoring.
"""
import os,math,copy
from pathlib import Path
from .scene import infer
from .layouts import elastic,shelves,hierarchy,constraint_search,transport
from .emit import realize,finalize
from .prior import retrieve
from .objective import score,pixel_cost
from .geometry import Rect,box,setbox,finite


def fit_source(source,target):
    W,H=target['width'],target['height'];W0=max(1,finite(source.get('canvas',{}).get('width'),1));H0=max(1,finite(source.get('canvas',{}).get('height'),1));sc=min(W/W0,H/H0);els=[]
    for e in source.get('elements',[]):
        o=copy.deepcopy(e);r=box(e)
        if e.get('type')!='text' and r.x<=0 and r.y<=0 and r.r>=W0 and r.b>=H0:
            setbox(o,Rect(0,0,W,H))
            if e['type']=='image':o['fit']='cover'
        else:
            setbox(o,Rect((W-W0*sc)/2+r.x*sc,(H-H0*sc)/2+r.y*sc,r.w*sc,r.h*sc))
            if e.get('type')=='text':o['font_size']=max(1,int(finite(e.get('font_size'),12)*sc))
            for key in ('corner_radius','stroke_width','letter_spacing'):
                if key in o:o[key]=finite(o[key])*sc
        els.append(o)
    return finalize(source,{'elements':els},W,H)


def candidates(source,target,method='portfolio'):
    W,H=int(target['width']),int(target['height']);s0=infer(source)
    baseline=fit_source(source,target);v,t=score(s0,baseline)
    yield {'name':'fit','output':baseline,'score':v,'terms':t}
    if len(source.get('elements',[]))>160:return
    for variant in ('relations','copy'):
        s=s0 if variant=='relations' else infer(source,variant)
        generators=[]
        if method in ('elastic','portfolio'):generators.extend([elastic(s,W,H),transport(s,W,H)])
        if method in ('hierarchy','portfolio'):generators.append(hierarchy(s,W,H))
        if method in ('search','portfolio'):generators.extend([constraint_search(s,W,H),hierarchy(s,W,H,True),shelves(s,W,H)])
        if method=='retrieval' or (method=='portfolio' and os.environ.get('ORIGIN_PRIOR','1')=='1'):generators.append(retrieve(s,W,H))
        for gen in generators:
            for name,cells in gen:
                for typo in (1.,1.4):
                    out=realize(s,cells,W,H,typo);v,t=score(s,out)
                    yield {'name':name+'/'+variant+f'/{typo}','output':out,'score':v,'terms':t}


def diagnose(source,target,method='portfolio',pixels=True):
    plans=sorted(candidates(source,target,method),key=lambda p:-p['score']);unique=[];seen=set()
    import json
    for p in plans:
        key=json.dumps(p['output'],sort_keys=True)
        if key not in seen:seen.add(key);unique.append(p)
    shortlist=unique[:4]
    if pixels:
        assets=os.environ.get('ORIGIN_ASSETS',str(Path(__file__).resolve().parents[1]/'brand_kit/assets'))
        for p in shortlist:
            try:
                penalty=pixel_cost(p['output'],assets);p['score']-=penalty;p['terms']['pixels']=-penalty
            except (ImportError,RuntimeError,ValueError,OSError):p['terms']['pixels_unavailable']=0
        shortlist.sort(key=lambda p:-p['score'])
    return shortlist[0],{'generated':len(plans),'unique':len(unique),'shortlist':[{k:v for k,v in p.items() if k!='output'} for p in shortlist]}


def resize(source,target_canvas):
    if source.get('canvas')==target_canvas and all(box(e).x>=0 and box(e).y>=0 and box(e).r<=target_canvas['width'] and box(e).b<=target_canvas['height'] for e in source.get('elements',[])):
        return copy.deepcopy(source)
    best,_=diagnose(source,target_canvas,os.environ.get('ORIGIN_METHOD','portfolio'),pixels=os.environ.get('ORIGIN_PIXELS','1')=='1')
    return best['output']


def diagnose_all(source,target,pixels=True):
    """One candidate pool, method ablations selected without regenerating it."""
    plans=list(candidates(source,target,'portfolio'));groups={
        'elastic':('fit','elastic','transport'),
        'hierarchy':('fit','hierarchy'),
        'search':('fit','constraint-search','partition','shelf','semantic-split','grid'),
        'retrieval':('fit','retrieval'),
        'portfolio':None}
    result={};assets=os.environ.get('ORIGIN_ASSETS',str(Path(__file__).resolve().parents[1]/'brand_kit/assets'))
    penalties={}
    import json
    for method,prefixes in groups.items():
        pool=sorted([p for p in plans if prefixes is None or p['name'].split('/')[0] in prefixes],key=lambda p:-p['score'])
        unique=[];seen=set()
        for p in pool:
            key=json.dumps(p['output'],sort_keys=True)
            if key in seen:continue
            seen.add(key);q=copy.deepcopy(p)
            if pixels and len(unique)<4:
                if key not in penalties:
                    try:penalties[key]=pixel_cost(p['output'],assets)
                    except (ImportError,RuntimeError,ValueError,OSError):penalties[key]=0
                q['score']-=penalties[key];q['terms']['pixels']=-penalties[key]
            unique.append(q)
            if len(unique)>=4:break
        result[method]=max(unique,key=lambda p:p['score'])
    return result,len(plans)
