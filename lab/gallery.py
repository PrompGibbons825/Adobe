import json,os,io,argparse
from pathlib import Path
from PIL import Image,ImageDraw
from renderer.converter.convert import _get_renderer,convert_chameleon_to_crello
from .experiment import tasks,WIDE,measures

def image(doc):
    renderer=_get_renderer(str(Path('brand_kit/fonts').resolve()));ex,_=convert_chameleon_to_crello(doc,renderer.font_manager,str(WIDE/'cases/assets'))
    return Image.open(io.BytesIO(renderer.render(ex,max_size=520,format='png'))).convert('RGB')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');p.add_argument('--count',type=int,default=8);a=p.parse_args()
    data=[json.loads(x) for x in (Path('results')/a.tag/'records.jsonl').read_text().splitlines()];srcs={n:(s,t,r) for n,s,t,r,f in tasks('all')};baselines={k:json.loads((WIDE/'bench'/k/'results.json').read_text()) for k in ('main','sprint')}
    data=[dict(name=r['name'],**r['methods']['portfolio']) if 'methods' in r else r for r in data]
    rows=[r for r in data if r['name'] in baselines['main'] and r['name'] in baselines['sprint']][:a.count]
    w,h=420,290;sheet=Image.new('RGB',(w*4,h*len(rows)+30),'#dedede');draw=ImageDraw.Draw(sheet)
    for k,label in enumerate(('Source','Main','Sprint','Origin')):draw.text((k*w+10,8),label,fill='black')
    for z,r in enumerate(rows):
        name=r['name'];src,_,_=srcs[name]
        docs=[src]+[baselines[k][name]['out'] for k in ('main','sprint')]+[r['output']]
        for k,doc in enumerate(docs):
            im=image(doc);im.thumbnail((w-16,h-48));x=k*w+(w-im.width)//2;y=z*h+36+(h-48-im.height)//2;sheet.paste(im,(x,y));draw.text((k*w+8,(z+1)*h+8),name[:12]+(' '+r['winner'] if k==3 else ''),fill='black')
    out=Path('results')/a.tag/'gallery.jpg';sheet.save(out,quality=90);print(out)
