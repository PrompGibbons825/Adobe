"""Render one source design at many formats into a contact sheet. python -m lab.sheet input/x.json -o out.png [--fmts a,b]"""
import argparse, json, sys, time
from pathlib import Path
from PIL import Image, ImageDraw
from engine.resize import resize
from engine.compose import compose
from renderer.converter.convert import render_design_to_png

from lab.sheet_fmt import FORMATS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("files",nargs="+"); ap.add_argument("-o",default="/tmp/sheet.png")
    ap.add_argument("--fmts"); ap.add_argument("--cell",type=int,default=320); a=ap.parse_args()
    fm=a.fmts.split(",") if a.fmts else list(FORMATS)
    tiles=[]; tmp=Path("/tmp/lab_tiles"); tmp.mkdir(exist_ok=True)
    for f in a.files:
        d=json.load(open(f)); src=d.get("source",d)
        for n in fm:
            W,H=FORMATS[n]; t0=time.time(); o=compose(src,{"width":W,"height":H}); dt=time.time()-t0
            lay=o.pop("_layout"); p=tmp/f"{Path(f).stem}_{n}.png"
            render_design_to_png(o,"brand_kit/assets",p,max_size=a.cell*2)
            im=Image.open(p).convert("RGB"); im.thumbnail((a.cell,a.cell))
            tile=Image.new("RGB",(a.cell,a.cell+16),(60,60,60)); tile.paste(im,((a.cell-im.width)//2,16))
            ImageDraw.Draw(tile).text((2,2),f"{Path(f).stem[-14:]} {n} {W}x{H} {lay} {dt:.1f}s",fill=(255,255,0))
            tiles.append(tile)
    cols=5; rows=(len(tiles)+cols-1)//cols
    sh=Image.new("RGB",(cols*a.cell,rows*(a.cell+16)))
    for i,t in enumerate(tiles): sh.paste(t,((i%cols)*a.cell,(i//cols)*(a.cell+16)))
    sh.save(a.o); print(a.o,len(tiles))
main()
