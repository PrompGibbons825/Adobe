"""Random designs x formats: hard rules + runtime. python -m lab.stress --n 100 --seed 1"""
import argparse, random, time, json
from engine.compose import compose
from lab.check import check
from lab.sheet_fmt import FORMATS

WORDS = "Chase the Light Summer Sale Limited Edition Botanical sparkling infusions crafted for golden hour Shop now Free shipping on all orders over today only".split()
def gen(rng):
    W, H = rng.choice([(1080,1080),(1080,1350),(1080,1920),(1200,628),(1600,900),(728,90),(300,250),(800,1200)])
    els = [dict(id="bg", type="image", x=0,y=0,width=W,height=H,z_index=0,asset_id="solstice_bg_gradient.png",fit="cover")]
    z = 1
    for i in range(rng.randint(1, 9)):
        t = rng.choice(["text","text","image","shape","group"])
        w = rng.uniform(.15,.9)*W; h = rng.uniform(.05,.4)*H; x = rng.uniform(0,W-w); y = rng.uniform(0,H-h)
        if t == "text":
            c = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1,14)))
            els.append(dict(id=f"t{i}",type="text",x=x,y=y,width=w,height=h,z_index=z,content=c,font_family=rng.choice(["Solstice Serif","Solstice Sans"]),
                 font_size=rng.uniform(.015,.07)*min(W,H)*rng.choice([1,1.5]),font_weight=rng.choice(["regular","bold"]),color="rgba(255,255,255,1)",text_align=rng.choice(["left","center","right"])))
        elif t == "image":
            els.append(dict(id=f"i{i}",type="image",x=x,y=y,width=w,height=h,z_index=z,asset_id=rng.choice(["solstice_bottle.png","solstice_logo_lockup.png","missing.png"]),fit="contain"))
        elif t == "shape":
            els.append(dict(id=f"s{i}",type="shape",x=x,y=y,width=w,height=h,z_index=z,shape_type=rng.choice(["rectangle","ellipse"]),fill_color="rgba(232,106,58,1)",opacity=rng.choice([1,.3])))
        else:
            els.append(dict(id=f"g{i}b",type="shape",x=x,y=y,width=w,height=h,z_index=z,shape_type="rectangle",fill_color="rgba(232,106,58,1)",group_id=f"g{i}"))
            els.append(dict(id=f"g{i}t",type="text",x=x,y=y+h*.2,width=w,height=h*.5,z_index=z+1,content="NEW "+rng.choice(WORDS),font_family="Solstice Sans",font_size=h*.3,color="rgba(255,255,255,1)",text_align="center",group_id=f"g{i}"))
        z += 2
    if rng.random() < .3: els[-1]["rotation"] = rng.uniform(-20,20)
    return {"canvas":{"width":W,"height":H},"elements":els}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=50); ap.add_argument("--seed",type=int,default=1); a = ap.parse_args()
    rng = random.Random(a.seed); bad = 0; tt = []; tot = 0
    for i in range(a.n):
        src = gen(rng)
        for name in rng.sample(list(FORMATS), 4):
            W, H = FORMATS[name]; t0 = time.time()
            try:
                out = compose(src, {"width": W, "height": H}); errs = check(src, out)
            except Exception as ex:
                errs = ["EXC " + repr(ex)]
            tt.append(time.time()-t0); tot += 1
            if errs: bad += 1; print("FAIL", i, name, errs[:4], flush=True)
    tt.sort(); print(f"{tot} runs, {bad} faulty, p50 {tt[len(tt)//2]:.2f}s p95 {tt[int(len(tt)*.95)]:.2f}s max {tt[-1]:.2f}s")
