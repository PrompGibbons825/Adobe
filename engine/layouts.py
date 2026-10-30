"""Independent composition generators: deformation, hierarchy and partition search.
Every generator returns cells for rigid islands; typography is realized later.
"""
import math
from .geometry import Rect,union


def elastic(s,W,H):
    # Empty space absorbs aspect change. Each island keeps its local metric.
    scale=min(W/s.W,H/s.H)
    for grow in (1.,1.3,1.7):
        cells={}
        for k,q in enumerate(s.islands):
            r=q.rect
            w=min(W*.94,r.w*scale*grow);h=min(H*.94,r.h*scale*grow)
            if q.text and not q.rigid:
                w=min(W*.9,max(w,r.w/s.W*W));h=min(H*.9,max(h,r.h/s.H*H))
            x=r.cx/s.W*W-w/2;y=r.cy/s.H*H-h/2
            cells[k]=Rect(max(0,min(W-w,x)),max(0,min(H-h,y)),w,h)
        yield 'elastic',cells


def _row(s,ids,r,axis,weights=None,gap=0):
    if not ids:return {}
    if weights is None:weights=[max(.01,(s.islands[k].rect.w if axis==0 else s.islands[k].rect.h)) for k in ids]
    avail=(r.w if axis==0 else r.h)-gap*(len(ids)-1);total=sum(weights);at=r.x if axis==0 else r.y;cells={}
    for k,weight in zip(ids,weights):
        z=max(.01,avail*weight/total)
        cells[k]=Rect(at,r.y,z,r.h) if axis==0 else Rect(r.x,at,r.w,z)
        at+=z+gap
    return cells


def shelves(s,W,H):
    ids=list(range(len(s.islands)));margin=min(W,H)*.045;R=Rect(0,0,W,H).inset(margin);gap=min(W,H)*.025
    for axis in (0,1):
        order=sorted(ids,key=lambda k: (s.islands[k].cx if axis==0 else s.islands[k].cy))
        for power in (.4,.7,1.):
            weights=[(max(1,s.islands[k].rect.w if axis==0 else s.islands[k].rect.h))**power for k in order]
            yield 'shelf',_row(s,order,R,axis,weights,gap)
    text=[k for k in ids if s.islands[k].text];art=[k for k in ids if not s.islands[k].text]
    if text and art:
        for axis in (0,1):
            for share in (.32,.45,.58):
                for flip in (False,True):
                    if axis==0:
                        first=Rect(R.x,R.y,(R.w-gap)*share,R.h);second=Rect(first.r+gap,R.y,R.w-first.w-gap,R.h)
                    else:
                        first=Rect(R.x,R.y,R.w,(R.h-gap)*share);second=Rect(R.x,first.b+gap,R.w,R.h-first.h-gap)
                    ra,rt=(second,first) if flip else (first,second)
                    cells=_row(s,art,ra,0 if ra.w>ra.h*2 else 1,gap=gap)
                    cells.update(_row(s,text,rt,0 if rt.w>rt.h*3 else 1,[s.islands[k].mass for k in text],gap))
                    yield 'semantic-split',cells
    # Grid is a first-class arrangement; bounded by sqrt(n), not factorial permutations.
    for cols in range(2,min(4,len(ids))+1):
        rows=math.ceil(len(ids)/cols);cells={}
        for z,k in enumerate(ids):
            cells[k]=Rect(R.x+(z%cols)*(R.w+gap)/cols,R.y+(z//cols)*(R.h+gap)/rows,(R.w-gap*(cols-1))/cols,(R.h-gap*(rows-1))/rows)
        yield 'grid',cells


def cuts(s,ids):
    """Source topology supplies meaningful separators, including imperfect ones."""
    choices=[]
    for axis in (0,1):
        order=sorted(ids,key=lambda k:s.islands[k].cx if axis==0 else s.islands[k].cy)
        for n in range(1,len(order)):
            left,right=order[:n],order[n:];a=union(s.islands[k].rect for k in left);b=union(s.islands[k].rect for k in right)
            overlap=a.inter(b)/max(1,min(a.area,b.area))
            gap=(b.x-a.r)/s.W if axis==0 else (b.y-a.b)/s.H
            # Penalize cutting across explicit visual structure, reward whitespace.
            choices.append((overlap*4-gap,axis,left,right))
    return sorted(choices,key=lambda x:x[0])[:3]


def hierarchy(s,W,H,search=False):
    ids=list(range(len(s.islands)));R=Rect(0,0,W,H).inset(min(W,H)*.04);gap=min(W,H)*.025
    if not ids:return
    # Multiple topology hypotheses, each with a deterministic bounded recursion.
    for mode in range(8 if search else 3):
        cells={}
        def divide(ids,r,depth):
            if len(ids)==1:cells[ids[0]]=r;return
            ch=cuts(s,ids);choice=ch[(mode//(depth+1))%len(ch)] if search else ch[0]
            _,source_axis,a,b=choice
            ra=union(s.islands[k].rect for k in a);rb=union(s.islands[k].rect for k in b)
            axis=source_axis
            if search:axis=0 if r.w/r.h>1.25 else 1
            elif mode==1:axis=0 if r.w/r.h>1.8 else 1
            wa=sum(s.islands[k].mass for k in a);wb=sum(s.islands[k].mass for k in b)
            srca=ra.w if axis==0 else ra.h;srcb=rb.w if axis==0 else rb.h
            share=(srca/(srca+srcb)+wa/(wa+wb))/2
            if search:share=.5*share+.5*(.35,.5,.65)[mode%3]
            share=max(.18,min(.82,share))
            if axis==0:
                z=max(.01,(r.w-gap)*share);ar=Rect(r.x,r.y,z,r.h);br=Rect(ar.r+gap,r.y,max(.01,r.w-z-gap),r.h)
            else:
                z=max(.01,(r.h-gap)*share);ar=Rect(r.x,r.y,r.w,z);br=Rect(r.x,ar.b+gap,r.w,max(.01,r.h-z-gap))
            divide(a,ar,depth+1);divide(b,br,depth+1)
        divide(ids,R,0)
        yield 'partition' if search else 'hierarchy',cells


def constraint_search(s,W,H):
    """Bounded best-first slicing search over source-informed graph separators.

    A state is (island set, target cell). Unlike a fixed stack/split catalogue,
    every internal node can change orientation and allocation. A budget bounds
    expansion; the incumbent is always a complete feasible floorplan.
    """
    from .typefit import fit,metrics
    from functools import lru_cache
    ids=tuple(range(len(s.islands)));gap=min(W,H)*.022;root=Rect(0,0,W,H).inset(min(W,H)*.04)
    if not ids or len(ids)>14:return
    visits=[0];limit=120
    # Upper-bound typography from total glyph area before exact final realization.
    widths={}
    for q in s.islands:
        for i in q.ids:
            e=s.els[i]
            if e["type"]=="text":
                advance,_,_=metrics(e,100,1e7);words=str(e.get("content","")).split();longest=max([len(w) for w in words] or [1]);chars=max(1,len(str(e.get("content",""))))
                widths[i]=(max(.01,advance/100),max(.01,advance/100*longest/chars))
    @lru_cache(maxsize=1024)
    def leaf(k,w,h):
        q=s.islands[k];cell=Rect(0,0,w,h);base=math.sqrt(W*H/(s.W*s.H));sc=min(w/max(.01,q.rect.w),h/max(.01,q.rect.h));value=0.
        if q.text:
            for i in q.ids:
                e=s.els[i]
                if e['type']!='text':continue
                fs0=max(1,float(e.get('font_size',12)));desired=min(fs0*base,min(W,H)*.32)
                total,word=widths[i]
                fs=min(desired,math.sqrt(max(.01,w*h)/(total*1.25)),w/word,h/1.25) if len(q.ids)==1 else max(1,int(fs0*sc))
                value+=math.log(max(.01,fs/(fs0*base)))*math.sqrt(fs0)
        else:value+=math.log(max(.01,sc/base))*q.mass*.45
        return value
    def quick(group,r):
        weights=[s.islands[k].mass for k in group];axis=0 if r.w>r.h*1.7 else 1
        cells=_row(s,group,r,axis,weights,gap)
        return sum(leaf(k,round(v.w,2),round(v.h,2)) for k,v in cells.items()),cells
    def solve(group,r,depth=0):
        visits[0]+=1
        if len(group)==1:return leaf(group[0],round(r.w,2),round(r.h,2)),{group[0]:r}
        incumbent=quick(group,r)
        if visits[0]>limit:return incumbent
        options=[]
        for cutcost,srcaxis,a,b in cuts(s,group)[:2]:
            wa=sum(s.islands[k].mass for k in a);wb=sum(s.islands[k].mass for k in b)
            for axis in (0,1):
                extent=r.w if axis==0 else r.h
                if extent<=gap*3:continue
                for share in sorted({.33,.5,.67,round(max(.2,min(.8,wa/(wa+wb))),2)}):
                    z=(extent-gap)*share
                    if axis==0:ar=Rect(r.x,r.y,z,r.h);br=Rect(ar.r+gap,r.y,r.w-z-gap,r.h)
                    else:ar=Rect(r.x,r.y,r.w,z);br=Rect(r.x,ar.b+gap,r.w,r.h-z-gap)
                    v1,_=quick(a,ar);v2,_=quick(b,br)
                    penalty=3*max(0,cutcost)+(1 if axis!=srcaxis else 0)
                    options.append((v1+v2-penalty,a,b,ar,br,penalty))
        # Beam orders promising allocations, then recursively refines only the best.
        for _,a,b,ar,br,penalty in sorted(options,key=lambda x:-x[0])[:3 if depth<2 else 2]:
            va,ca=solve(tuple(a),ar,depth+1);vb,cb=solve(tuple(b),br,depth+1);v=va+vb-penalty
            if v>incumbent[0]:incumbent=v,{**ca,**cb}
        return incumbent
    _,cells=solve(ids,root)
    yield 'constraint-search',cells


def transport(s,W,H):
    """Two one-dimensional monotone transport fields with rigid island springs.

    Most stretching is assigned to whitespace intervals. Endpoints span the
    target canvas, while protected intervals resist anisotropic deformation.
    This is an independent continuous alternative to the slicing search.
    """
    import numpy as np
    if not s.islands:return
    base=min(W/s.W,H/s.H)
    for tension in (.25,1.,4.):
        fields=[]
        for axis,target,source_extent in ((0,W,s.W),(1,H,s.H)):
            points=sorted({0.,float(source_extent)}|{max(0.,min(source_extent,v)) for q in s.islands for v in ((q.rect.x,q.rect.r) if axis==0 else (q.rect.y,q.rect.b))})
            n=len(points);A=[];b=[];p=np.array(points)
            # Local slopes have a uniform prior; protected island spans want a rigid scale.
            for j in range(n-1):
                row=np.zeros(n);row[j]=-1;row[j+1]=1;A.append(row*.25);b.append((p[j+1]-p[j])*target/source_extent*.25)
            for q in s.islands:
                a,z=(q.rect.x,q.rect.r) if axis==0 else (q.rect.y,q.rect.b)
                i=min(range(n),key=lambda j:abs(points[j]-a));j=min(range(n),key=lambda j:abs(points[j]-z))
                if i==j:continue
                row=np.zeros(n);row[i]=-1;row[j]=1
                weight=math.sqrt(max(1,q.mass))*tension
                A.append(row*weight);b.append((z-a)*base*weight)
            row=np.zeros(n);row[0]=1000;A.append(row);b.append(0)
            row=np.zeros(n);row[-1]=1000;A.append(row);b.append(target*1000)
            x=np.linalg.lstsq(np.array(A),np.array(b),rcond=None)[0]
            # Isotonic projection ensures the field cannot reverse reading order.
            x=np.maximum.accumulate(np.clip(x,0,target));x[0]=0;x[-1]=target
            fields.append((p,x))
        cells={}
        for k,q in enumerate(s.islands):
            r=q.rect;x0,x1=np.interp([r.x,r.r],*fields[0]);y0,y1=np.interp([r.y,r.b],*fields[1])
            cells[k]=Rect(float(x0),float(y0),max(1.,float(x1-x0)),max(1.,float(y1-y0)))
        yield 'transport',cells
