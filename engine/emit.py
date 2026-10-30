"""Realize a scene into cells, then enforce the input/output contract once.
No visibility changes, no copy rewriting, no asset pixel synthesis.
"""
import copy,math
from .geometry import Rect,box,setbox,finite,rotated
from .typefit import fit,metrics


def transform(e,src,dst,scale):
    o=copy.deepcopy(e);r=box(e)
    setbox(o,Rect(dst.x+(r.x-src.x)*scale,dst.y+(r.y-src.y)*scale,r.w*scale,r.h*scale))
    for key in ('corner_radius','stroke_width','letter_spacing'):
        if key in o:o[key]=finite(o[key])*scale
    if o.get('type')=='text':o['font_size']=max(1,int(finite(o.get('font_size'),12)*scale))
    return o


def realize(s,cells,W,H,typography=1.):
    out={};base=math.sqrt(W*H/(s.W*s.H));short=min(W,H)
    for i in s.bg:
        e=copy.deepcopy(s.els[i]);setbox(e,Rect(0,0,W,H))
        if e['type']=='image':e['fit']='cover'
        out[i]=e
    for k,q in enumerate(s.islands):
        cell=cells[k];src=q.rect
        if q.panel and sum(s.els[i]['type']=='text' for i in q.ids)==1:
            label=next(i for i in q.ids if s.els[i]['type']=='text');e=copy.deepcopy(s.els[label]);fs0=finite(e.get('font_size'),12)
            pad=min(short*.015,cell.w*.07,cell.h*.15);inside=cell.inset(pad)
            fs,ink,h,n=fit(e,inside,min(fs0*base*typography,short*.2))
            bw=min(cell.w,max(ink+2*pad,cell.w*.4));bh=min(cell.h,h+2*pad);panel=Rect(cell.cx-bw/2,cell.cy-bh/2,bw,bh)
            for i in q.ids:
                if i==label:continue
                o=copy.deepcopy(s.els[i]);setbox(o,panel)
                if 'corner_radius' in o:o['corner_radius']=min(bh/2,finite(o['corner_radius'])*base)
                out[i]=o
            e['font_size']=fs;setbox(e,Rect(panel.x+pad,panel.y+pad,max(1,panel.w-2*pad),h));out[label]=e
        elif q.text and len(q.ids)==1:
            i=q.ids[0];e=copy.deepcopy(s.els[i]);fs0=finite(e.get('font_size'),12)
            desired=min(fs0*base*typography,max(12,short*.21))
            fs,ink,h,n=fit(e,cell,desired)
            w=cell.w
            # Text blocks occupy their real height; source alignment is kept.
            e['font_size']=fs
            if 'letter_spacing' in e:e['letter_spacing']=finite(e['letter_spacing'])*fs/max(1,fs0)
            setbox(e,Rect(cell.x,cell.cy-h/2,w,h));out[i]=e
        elif q.text and not q.rigid:
            # Reflow a typographic group with its own hierarchy and source rhythm.
            ids=sorted(q.ids,key=lambda i:box(s.els[i]).cy)
            raw=[max(1,finite(s.els[i].get('font_size'),12)) for i in ids]
            gap=min(cell.h*.035,short*.025)
            lo,hi=.01,base*typography*2
            best=None
            for _ in range(12):
                sc=(lo+hi)/2;sizes=[max(1,int(v*sc)) for v in raw]
                dims=[metrics(s.els[i],fs,cell.w) for i,fs in zip(ids,sizes)]
                height=sum(d[1] for d in dims)+gap*(len(ids)-1)
                if height<=cell.h and all(d[0]<=cell.w for d in dims):best=(sizes,dims,height);lo=sc
                else:hi=sc
            if best is None:best=([1]*len(ids),[metrics(s.els[i],1,cell.w) for i in ids],sum(metrics(s.els[i],1,cell.w)[1] for i in ids))
            sizes,dims,height=best;y=cell.cy-height/2
            for i,fs,(_,h,_) in zip(ids,sizes,dims):
                e=copy.deepcopy(s.els[i]);e['font_size']=fs
                if 'letter_spacing' in e:e['letter_spacing']=finite(e['letter_spacing'])*fs/max(1,finite(s.els[i].get('font_size'),12))
                setbox(e,Rect(cell.x,y,cell.w,h));out[i]=e;y+=h+gap
        else:
            sc=min(cell.w/max(.01,src.w),cell.h/max(.01,src.h))
            dst=Rect(cell.cx-src.w*sc/2,cell.cy-src.h*sc/2,src.w*sc,src.h*sc)
            for i in q.ids:out[i]=transform(s.els[i],src,dst,sc)
    # Ornamental islands move with their nearest canvas boundary. Their internal
    # arrangement is preserved; the blank central region remains available to copy.
    from .geometry import union
    content=union(box(e) for i,e in out.items() if i not in s.bg)
    for ids,src in s.decor_groups:
        sc=min(W/s.W,H/s.H)*.9
        w,h=src.w*sc,src.h*sc
        # Source edge distance, rather than individual object centers, is the anchor.
        x=src.x/s.W*W if src.cx<s.W/2 else W-(s.W-src.r)/s.W*W-w
        y=src.y/s.H*H if src.cy<s.H/2 else H-(s.H-src.b)/s.H*H-h
        dst=Rect(max(0,min(W-w,x)),max(0,min(H-h,y)),w,h)
        # Keep frame pieces out of the central copy when aspect changes collapse the moat.
        if dst.inter(content)>.12*dst.area:
            trials=[Rect(0,dst.y,w,h),Rect(W-w,dst.y,w,h),Rect(dst.x,0,w,h),Rect(dst.x,H-h,w,h)]
            dst=min(trials,key=lambda r:r.inter(content)+.05*((r.cx-dst.cx)**2+(r.cy-dst.cy)**2))
        for i in ids:out[i]=transform(s.els[i],src,dst,sc)
    return finalize(s.source,{'canvas':{'width':int(W),'height':int(H)},'elements':[out.get(i,copy.deepcopy(e)) for i,e in enumerate(s.els)]},W,H)


def finalize(source,out,W,H):
    result=[]
    for se,e in zip(source['elements'],out['elements']):
        e=copy.deepcopy(e)
        for key in ('id','type','content','font_family','asset_id','color','fill_color','style_runs','rotation','font_weight','font_style','group_id','opacity','z_index'):
            if key in se:e[key]=se[key]
        r=box(e);w,h=max(.001,r.w),max(.001,r.h)
        if e.get('type')=='image' and not (e.get('fit')=='cover' and w>=W-.5 and h>=H-.5):
            sw,sh=max(.001,finite(se.get('width'),1)),max(.001,finite(se.get('height'),1));ratio=sw/sh
            w=min(w,h*ratio);h=w/ratio
        rr=rotated(Rect(0,0,w,h),finite(e.get('rotation')))
        sc=min(1,W/max(.001,rr.w),H/max(.001,rr.h));w*=sc;h*=sc
        x=r.cx-w/2;y=r.cy-h/2;rr=rotated(Rect(x,y,w,h),finite(e.get('rotation')))
        x+=max(0,-rr.x)-max(0,rr.r-W);y+=max(0,-rr.y)-max(0,rr.b-H)
        setbox(e,Rect(x,y,w,h))
        if e.get('type')=='text':
            # The renderer ignores box height. Fit the painted text, not just its JSON box.
            fs,_,_,_=fit(e,Rect(x,y,w,h),max(1,int(finite(e.get('font_size'),12)*sc)))
            e['font_size']=fs
        for key in ('x','y','width','height'):e[key]=round(e[key],5)
        result.append(e)
    return {'canvas':{'width':int(W),'height':int(H)},'elements':result}
