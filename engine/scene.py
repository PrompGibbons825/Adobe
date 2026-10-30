"""Infer scene relationships, keeping structural and ornamental graphs separate.

Explicit groups, contained labels and intersecting artwork form rigid islands.
Peripheral artwork is a frame around negative space, never a product list.
Ambiguous text grouping is represented by multiple scene hypotheses.
"""
from dataclasses import dataclass,field
import copy,math,os
from pathlib import Path
from functools import lru_cache
from .geometry import Rect,box,union,finite
from .typefit import metrics,footprint,painted

@lru_cache(maxsize=256)
def asset_stats(asset):
    from PIL import Image,ImageStat
    root=Path(os.environ.get('ORIGIN_ASSETS',str(Path(__file__).resolve().parents[1]/'brand_kit/assets')))
    try:
        with Image.open(root/asset) as im:
            im=im.convert('RGBA');im.thumbnail((64,64));a=im.getchannel('A');bb=a.getbbox()
            v=ImageStat.Stat(im.convert('RGB')).stddev
            return min(ImageStat.Stat(a).mean)/255, max(v), bb is not None and bb!=(0,0,im.width,im.height)
    except Exception:return 1.,100.,False

@dataclass
class Island:
    ids: tuple
    rect: Rect
    text: bool
    mass: float
    rigid: bool=False
    panel: bool=False
    @property
    def cx(self): return self.rect.cx
    @property
    def cy(self): return self.rect.cy

@dataclass
class Scene:
    source: dict
    W:float
    H:float
    els:list
    bg:list
    decor:list
    islands:list
    variant:str
    permitted:set=field(default_factory=set)
    decor_groups:list=field(default_factory=list)

class DSU:
    def __init__(self,n):self.p=list(range(n))
    def find(self,i):
        while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
        return i
    def join(self,i,j):self.p[self.find(i)]=self.find(j)


def infer(source,variant='relations'):
    source=copy.deepcopy(source);W=max(1,finite(source.get('canvas',{}).get('width'),1));H=max(1,finite(source.get('canvas',{}).get('height'),1))
    els=source.get('elements',[]);n=len(els);canvas=Rect(0,0,W,H);texts=[i for i,e in enumerate(els) if e.get('type')=='text' and str(e.get('content','')).strip()]
    bg=[];decor=[];ds=DSU(n);panels=set();permitted=set()
    ornament={i for i,e in enumerate(els) if e.get('type')=='shape' and not e.get('group_id') and finite(e.get('opacity'),1)<.45}
    for i,e in enumerate(els):
        r=box(e)
        if e.get('type')!='text' and r.inter(canvas)>=.96*canvas.area and e.get('z_index',0)<=min([els[t].get('z_index',0) for t in texts] or [1]):bg.append(i)
    for i,e in enumerate(els):
        if i in bg or i in ornament:continue
        for j in range(i):
            if j in bg or j in ornament:continue
            f=els[j];r,s=box(e),box(f);inter=r.inter(s)
            if i in texts and j in texts:
                ri,rj=painted(e),painted(f)
                if ri.inter(rj)>.08*max(.001,min(ri.area,rj.area)):permitted.add((min(i,j),max(i,j)))
                fs=max(finite(e.get('font_size'),12),finite(f.get('font_size'),12))
                gap=max(ri.x-rj.r,rj.x-ri.r)
                if abs(finite(e.get('rotation'))-finite(f.get('rotation')))<.1 and abs(ri.y-rj.y)<.22*fs and -.1*fs<=gap<1.5*fs:ds.join(i,j)
            elif inter>.08*max(.001,min(r.area,s.area)):permitted.add((min(i,j),max(i,j)))
            if e.get('group_id') is not None and e.get('group_id')==f.get('group_id'):ds.join(i,j)
            # Label on a non-text object; a large photo is a composited island only for short overlays.
            if (i in texts)!=(j in texts):
                t,p=(i,j) if i in texts else (j,i);rt,rp=box(els[t]),box(els[p]);pe=els[p]
                if rp.inter(rt)>=.85*rt.area and pe.get('z_index',0)<els[t].get('z_index',0):
                    if pe.get('type')=='shape' and pe.get('shape_type')!='ellipse':ds.join(t,p);panels.add(p)
                    elif pe.get('type')=='image' and rp.area<.65*canvas.area and rt.area<.35*rp.area:ds.join(t,p)
            # Overlapping non-text pieces are one illustration, not separate semantic roles.
            elif i not in texts and j not in texts and inter>.035*max(.001,min(r.area,s.area)):ds.join(i,j)
    groups={}
    for i in range(n):
        if i not in bg and i not in ornament:groups.setdefault(ds.find(i),[]).append(i)
    # A frame consists of multiple small peripheral illustrations around central copy.
    nongroups=[g for g in groups.values() if not any(i in texts for i in g)]
    text_u=union(box(els[i]) for i in texts)
    many_art=len(nongroups)>=4
    for g in nongroups:
        r=union(box(els[i]) for i in g);outside=not (text_u.x<r.cx<text_u.r and text_u.y<r.cy<text_u.b)
        small=r.area<.2*canvas.area
        shape=all(els[i].get('type')=='shape' for i in g)
        translucent=all(finite(els[i].get('opacity'),1)<.45 for i in g)
        edge=r.x<.06*W or r.y<.06*H or r.r>.94*W or r.b>.94*H
        if (many_art and small and outside) or (shape and (edge or translucent)):
            decor.extend(g)
    decor.extend(sorted(ornament))
    islands=[]
    for g in groups.values():
        g=[i for i in g if i not in decor]
        if not g:continue
        ts=[i for i in g if i in texts];r=union(box(els[i]) for i in g)
        mass=sum((max(4,finite(els[i].get('font_size'),12))**1.3 * max(1,len(str(els[i].get('content','')).split()))**.35) if i in texts else math.sqrt(max(1,box(els[i]).area))*.3 for i in g)
        islands.append(Island(tuple(g),r,bool(ts),max(1,mass),len(g)>1,any(i in panels for i in g)))
    if variant=='copy':
        # Keep the typography as a unit. Panels and text-on-art stay separate.
        free=[q for q in islands if q.text and len(q.ids)==1 and not els[q.ids[0]].get('rotation')]
        if len(free)>1:
            # Only nearby type shares a reflow block; distant legal/corner copy
            # must not steal the headline's line length.
            free.sort(key=lambda q:q.rect.y);blocks=[]
            for q in free:
                if blocks:
                    prev=blocks[-1][-1];gap=q.rect.y-prev.rect.b
                    a,b=els[prev.ids[0]],els[q.ids[0]]
                    aligned=abs(q.cx-prev.cx)<.1*W or abs(q.rect.x-prev.rect.x)<.05*W
                    if aligned and gap<1.2*min(prev.rect.h,q.rect.h) and len(blocks[-1])<3:
                        blocks[-1].append(q);continue
                blocks.append([q])
            for block in blocks:
                if len(block)<2:continue
                ids=tuple(i for q in block for i in q.ids)
                islands=[q for q in islands if q not in block]+[Island(ids,union(q.rect for q in block),True,sum(q.mass for q in block),False)]
    islands.sort(key=lambda q:(round(q.cy/H,2),q.cx/W,tuple(str(els[i].get('id')) for i in q.ids)))
    # Compile the ornamental graph once, not once per candidate.
    dd=DSU(n)
    for zi,i in enumerate(decor):
        for j in decor[:zi]:
            if box(els[i]).inter(box(els[j]))>0:dd.join(i,j)
    dg={}
    for i in decor:dg.setdefault(dd.find(i),[]).append(i)
    decor_groups=[(ids,union(box(els[i]) for i in ids)) for ids in dg.values()]
    return Scene(source,W,H,els,bg,decor,islands,variant,permitted,decor_groups)
