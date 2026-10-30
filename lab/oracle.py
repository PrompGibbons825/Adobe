"""Visual candidate audit: show alternatives before changing the generator/scorer."""
import json,os,argparse
from pathlib import Path
from PIL import Image,ImageDraw
from engine.resize import candidates
from .experiment import tasks,WIDE
from .gallery import image
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();os.environ['ORIGIN_ASSETS']=str(WIDE/'cases/assets');source,target=next((s,t) for n,s,t,r,f in tasks('all') if n==a.name)
plans=sorted(candidates(source,target),key=lambda p:-p['score']);seen=set();picked=[]
for plan in plans:
    kind=plan['name'].split('/')[0]
    if kind not in seen:seen.add(kind);picked.append(plan)
    if len(picked)>=9:break
sheet=Image.new('RGB',(1200,320*3),'#dddddd');draw=ImageDraw.Draw(sheet)
for z,p in enumerate(picked):
    im=image(p['output']);im.thumbnail((380,275));x=(z%3)*400+(400-im.width)//2;y=(z//3)*320+30;sheet.paste(im,(x,y));draw.text(((z%3)*400+8,(z//3)*320+8),p['name']+' '+str(round(p['score'],1)),fill='black')
out=Path('results')/'oracle';out.mkdir(exist_ok=True);sheet.save(out/(a.name+'.jpg'));print(out/(a.name+'.jpg'))
