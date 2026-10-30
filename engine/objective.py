"""A single quality objective on finalized output; feasibility stays separate.
Weights are fixed before the held-out comparison. Pixel terms see exactly what
will be emitted, with copy removed to measure the ground under the glyphs.
"""
import math,re,io
from .geometry import box,rotated,finite
from .typefit import footprint,metrics,painted


def violations(source,out):
    errors=[];W,H=out['canvas']['width'],out['canvas']['height'];src={e['id']:e for e in source['elements']};seen=[]
    for e in out['elements']:
        seen.append(e['id']);r=rotated(box(e),finite(e.get('rotation')))
        if r.x<-.01 or r.y<-.01 or r.r>W+.01 or r.b>H+.01:errors.append('bounds:'+str(e['id']))
        for key in ('type','content','font_family','asset_id','style_runs','rotation','opacity'):
            if key in src[e['id']] and e.get(key)!=src[e['id']][key]:errors.append('changed:'+key)
        if e['type']=='image' and not(e.get('fit')=='cover' and box(e).w>=W-.5 and box(e).h>=H-.5):
            a,b=box(e),box(src[e['id']])
            if b.w>0 and b.h>0 and abs((a.w/a.h)/(b.w/b.h)-1)>.05:errors.append('ratio:'+e['id'])
        if e['type']=='text':
            r=painted(e)
            if r.x<-.5 or r.y<-.5 or r.r>W+.5 or r.b>H+.5:errors.append('glyph:'+str(e['id']))
    if len(seen)!=len(set(seen)) or set(seen)!=set(src):errors.append('ids')
    return errors


def score(s,out):
    els=out['elements'];W,H=out['canvas']['width'],out['canvas']['height'];area=W*H;sc=math.sqrt(area/(s.W*s.H))
    texts=[i for i,e in enumerate(els) if e['type']=='text' and str(e.get('content','')).strip() and finite(e.get('opacity'),1)>0]
    bounds={i:painted(els[i]) for i in texts}
    terms={'feasible':-10000*len(violations(s.source,out)),'readability':0.,'collision':0.,'hierarchy':0.,'fidelity':0.,'composition':0.}
    floor=max(9,10*max(W,H)/1024)
    for i in texts:
        e=els[i];fs=e['font_size'];old=max(1,finite(s.els[i].get('font_size'),12));weight=math.sqrt(old/max(1,max(finite(s.els[j].get('font_size'),12) for j in texts)))
        # Log utility gives copy size diminishing returns, with a steep unreadability charge.
        desired=max(floor,min(old*sc,min(W,H)*.3))
        terms['readability']+=weight*(24*math.log(max(.01,fs/desired))-45*max(0,1-fs/floor))
        _,_,lines=metrics(e,fs,e['width'])
        source_lines=metrics(s.els[i],int(old),s.els[i]['width'])[2]
        terms['composition']-=max(0,lines-4)*2*weight+max(0,lines-source_lines)*10*weight
        for j in texts:
            if j>=i:continue
            if (j,i) not in s.permitted:
                inter=bounds[i].inter(bounds[j]);terms['collision']-=10000*inter/max(1,min(bounds[i].area,bounds[j].area))
            f0=max(1,finite(s.els[j].get('font_size'),12));f=els[j]['font_size']
            # Preserve ordering, allow display amplification. Equal-sized copy should stay equal.
            ratio=(fs/max(1,f))/(old/f0)
            terms['hierarchy']-=5*abs(math.log(max(.01,ratio)))
        for j,f in enumerate(els):
            if f['type']=='text' or j in s.bg:continue
            if (min(i,j),max(i,j)) in s.permitted:continue
            inter=bounds[i].inter(box(f))
            if inter and finite(f.get('opacity'),1)>.4:terms['collision']-=85*inter/max(1,bounds[i].area)
    for q in s.islands:
        if not q.text:
            for i in q.ids:
                src=box(s.els[i]);r=box(els[i]);want=min(.35,src.area/(s.W*s.H))*area
                terms['fidelity']-=30*max(0,math.log(max(1,want)/max(1,r.area)))
    # Alignments that existed across distinct text blocks are relationships, not labels.
    aspect=abs(math.log((W/H)/(s.W/s.H)))
    for z,i in enumerate(texts):
        for j in texts[:z]:
            a,b=box(s.els[i]),box(s.els[j]);c,d=box(els[i]),box(els[j])
            if abs(a.cx-b.cx)<.015*s.W:
                terms['composition']-=min(12,30*abs(c.cx-d.cx)/W)/(1+aspect)
    if texts:
        total=sum(bounds[i].area for i in texts);cx=sum(bounds[i].cx*bounds[i].area for i in texts)/max(1,total);cy=sum(bounds[i].cy*bounds[i].area for i in texts)/max(1,total)
        # Keep the source copy's intentional offset, softened toward the center.
        before=[box(s.els[i]) for i in texts];oldarea=sum(r.area for r in before)
        ox=sum(r.cx*r.area for r in before)/max(1,oldarea)/s.W;oy=sum(r.cy*r.area for r in before)/max(1,oldarea)/s.H
        terms['composition']-=10*(abs(cx/W-(.5+ox)/2)+abs(cy/H-(.5+oy)/2))
    return sum(terms.values()),terms


from functools import lru_cache
import json
@lru_cache(maxsize=12)
def ground_pixels(serialized,assets):
    from .appearance import ground
    import numpy as np
    im=ground(json.loads(serialized),assets);arr=np.asarray(im,dtype=float)/255
    lum=.2126*arr[:,:,0]+.7152*arr[:,:,1]+.0722*arr[:,:,2]
    return lum,im.width,im.height

def pixel_cost(out,assets):
    import numpy as np
    ground={'canvas':out['canvas'],'elements':[e for e in out['elements'] if e['type']!='text']}
    lum,iw,ih=ground_pixels(json.dumps(ground,sort_keys=True),str(assets))
    W,H=out['canvas']['width'],out['canvas']['height'];cost=0
    for e in out['elements']:
        if e['type']!='text' or not str(e.get('content','')).strip():continue
        r=footprint(e);x0=max(0,int(r.x/W*iw));x1=min(iw,max(x0+1,math.ceil(r.r/W*iw)));y0=max(0,int(r.y/H*ih));y1=min(ih,max(y0+1,math.ceil(r.b/H*ih)))
        patch=lum[y0:y1,x0:x1]
        if not patch.size:continue
        nums=re.findall(r'[\d.]+',str(e.get('color','rgba(0,0,0,1)')))
        if len(nums)<3:continue
        c=sum(float(nums[k])/255*v for k,v in enumerate((.2126,.7152,.0722)))
        contrast=(np.maximum(patch,c)+.05)/(np.minimum(patch,c)+.05)
        cost+=12*float(np.mean(np.maximum(0,1-contrast/3)))+8*float(np.std(patch))
    return cost
