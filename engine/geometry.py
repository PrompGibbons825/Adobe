"""Small, engine-independent rectangle primitives. Coordinates remain floating point."""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    @property
    def r(self): return self.x+self.w
    @property
    def b(self): return self.y+self.h
    @property
    def cx(self): return self.x+self.w/2
    @property
    def cy(self): return self.y+self.h/2
    @property
    def area(self): return max(0,self.w)*max(0,self.h)
    def inter(self,o): return max(0,min(self.r,o.r)-max(self.x,o.x))*max(0,min(self.b,o.b)-max(self.y,o.y))
    def union(self,o): return Rect(min(self.x,o.x),min(self.y,o.y),max(self.r,o.r)-min(self.x,o.x),max(self.b,o.b)-min(self.y,o.y))
    def inset(self,p): return Rect(self.x+p,self.y+p,max(.01,self.w-2*p),max(.01,self.h-2*p))

def finite(v,d=0):
    try:
        x=float(v)
        return x if math.isfinite(x) else d
    except (TypeError,ValueError): return d

def box(e): return Rect(*(finite(e.get(k)) for k in ('x','y','width','height')))
def union(rs):
    rs=list(rs)
    if not rs:return Rect(0,0,1,1)
    r=rs[0]
    for q in rs[1:]:r=r.union(q)
    return r

def rotated(r,angle):
    a=math.radians(angle); c,s=abs(math.cos(a)),abs(math.sin(a))
    w,h=c*r.w+s*r.h,s*r.w+c*r.h
    return Rect(r.cx-w/2,r.cy-h/2,w,h)

def setbox(e,r):
    e.update(x=r.x,y=r.y,width=max(.001,r.w),height=max(.001,r.h)); return e
