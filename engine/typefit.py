"""Text metrics from the official converter, with bounded immutable caches.
No engine implementation or engine-specific measuring helpers are imported.
"""
import os
from functools import lru_cache
from pathlib import Path
from .geometry import Rect,finite

ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=4)
def manager():
    from renderer.converter.brand_fonts import load_brand_font_manager
    return load_brand_font_manager(str(ROOT/'brand_kit/fonts'))

@lru_cache(maxsize=512)
def advances(family,weight,style,text):
    from renderer.converter.wrap import char_widths
    return tuple(char_widths(manager(),family,weight,style,100.0,text))

def metrics(e,fs,width):
    from renderer.converter.wrap import wrap_text
    text=str(e.get('content',''))
    weight='bold' if str(e.get('font_weight','regular')).lower() in ('bold','semibold','black') else 'regular'
    italic=e.get('font_style')=='italic'
    style=('bolditalic' if weight=='bold' else 'italic') if italic else ('bold' if weight=='bold' else 'regular')
    try: ws=[v*fs/100 for v in advances(str(e.get('font_family','Solstice Sans')),weight,style,text)]
    except Exception: ws=[fs*.56 for c in text]
    wrapped=wrap_text(text,ws,max(.01,width))
    n=1; line=0.; maxline=0.; chars=0; track=finite(e.get('letter_spacing'))*fs/max(1,finite(e.get('font_size'),12))
    for c,w in zip(wrapped,ws):
        if c=='\n':maxline=max(maxline,line+max(0,chars-1)*track);line=0;chars=0;n+=1
        else:line+=w;chars+=1
    maxline=max(maxline,line+max(0,chars-1)*track)
    # cr-renderer paints baseline at top + font size; allow descenders and styled runs.
    height=fs*(1.25+(n-1)*max(.5,finite(e.get('line_height'),1.2)))
    return maxline,height,n

def fit(e,r,desired,minfs=1):
    hi=max(minfs,int(desired));lo=minfs;best=None
    for _ in range(14):
        if lo>hi:break
        fs=(lo+hi)//2;w,h,n=metrics(e,fs,r.w)
        if w<=r.w+.05 and h<=r.h+.05:best=(fs,w,h,n);lo=fs+1
        else:hi=fs-1
    if best is None:
        w,h,n=metrics(e,minfs,r.w);best=(minfs,w,h,n)
    return best

def footprint(e):
    r=Rect(e['x'],e['y'],e['width'],e['height'])
    w,h,_=metrics(e,int(e.get('font_size',12)),r.w)
    align=e.get('text_align','left')
    x=r.x+({'center':.5,'right':1}.get(align,0))*(r.w-w)
    return Rect(x,r.y,w,h)


def painted(e):
    """Glyph AABB after rotation about the *text box* center, as cr-renderer does."""
    import math
    from .geometry import box
    r=footprint(e);b=box(e);a=math.radians(finite(e.get('rotation')));c,s=math.cos(a),math.sin(a);points=[]
    for x,y in ((r.x,r.y),(r.r,r.y),(r.r,r.b),(r.x,r.b)):
        dx,dy=x-b.cx,y-b.cy;points.append((b.cx+c*dx-s*dy,b.cy+s*dx+c*dy))
    xs,ys=zip(*points);return Rect(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
