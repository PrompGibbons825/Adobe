"""Bounded low-resolution compositor for *scoring only*.
Original asset files and emitted asset references are never modified. Thumbnail
caches avoid decoding megapixel assets once per candidate. Final previews and
submission outputs always use the untouched official renderer.
"""
import math,os,re
from pathlib import Path
from functools import lru_cache
from PIL import Image,ImageColor,ImageDraw

def rgba(value):
    s=str(value or 'rgba(0,0,0,0)');nums=re.findall(r'[-\d.]+',s)
    if s.startswith('rgb') and len(nums)>=3:return tuple(max(0,min(255,int(float(v)))) for v in nums[:3])+(int(float(nums[3])*255) if len(nums)>3 else 255,)
    try:return ImageColor.getcolor(s,'RGBA')
    except (ValueError,TypeError):return (0,0,0,0)

@lru_cache(maxsize=96)
def thumbnail(path):
    try:
        with Image.open(path) as src:
            src=src.convert('RGBA');src.thumbnail((256,256));return src.copy()
    except (FileNotFoundError,OSError):return Image.new('RGBA',(1,1),(150,150,150,255))

def ground(out,assets):
    W,H=out['canvas']['width'],out['canvas']['height'];sc=min(1,192/max(W,H));iw,ih=max(1,round(W*sc)),max(1,round(H*sc));im=Image.new('RGBA',(iw,ih),(255,255,255,255))
    for e in sorted(out['elements'],key=lambda e:e.get('z_index',0)):
        if e['type']=='text':continue
        x,y=round(e['x']*sc),round(e['y']*sc);w,h=max(1,round(e['width']*sc)),max(1,round(e['height']*sc))
        if e['type']=='image':
            src=thumbnail(str(Path(assets)/str(e.get('asset_id',''))));fit=e.get('fit','contain')
            if fit=='fill':tile=src.resize((w,h),Image.Resampling.BILINEAR)
            else:
                factor=(max if fit=='cover' else min)(w/src.width,h/src.height);nw,nh=max(1,round(src.width*factor)),max(1,round(src.height*factor));pic=src.resize((nw,nh),Image.Resampling.BILINEAR)
                if fit=='cover':tile=pic.crop(((nw-w)//2,(nh-h)//2,(nw+w)//2,(nh+h)//2))
                else:tile=Image.new('RGBA',(w,h));tile.alpha_composite(pic,((w-nw)//2,(h-nh)//2))
        else:
            tile=Image.new('RGBA',(w,h));draw=ImageDraw.Draw(tile);col=rgba(e.get('fill_color'));kind=e.get('shape_type','rectangle')
            if kind=='ellipse':draw.ellipse((0,0,w-1,h-1),fill=col)
            elif kind=='line':draw.line((0,h//2,w,h//2),fill=col,width=max(1,round(e.get('stroke_width',1)*sc)))
            else:draw.rounded_rectangle((0,0,w-1,h-1),radius=max(0,float(e.get('corner_radius',0))*sc),fill=col)
        opacity=max(0,min(1,float(e.get('opacity',1))))
        if opacity<1:tile.putalpha(tile.getchannel('A').point(lambda v:int(v*opacity)))
        if e.get('rotation'):
            tile=tile.rotate(-float(e['rotation']),resample=Image.Resampling.BICUBIC,expand=True);x-=round((tile.width-w)/2);y-=round((tile.height-h)/2)
        im.alpha_composite(tile,(x,y))
    return im.convert('RGB')
