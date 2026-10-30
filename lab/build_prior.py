"""Compile cross-family geometry priors; exclude every evaluation family.
Raw designer coordinates only. No incumbent code, images, labels or panel data.
"""
import json,math,hashlib
from pathlib import Path
WIDE=Path('/data/agents/work/crello-wide')
def normalized_text(x):return ' '.join(str(x).lower().split())
def compact(m):
    W,H=m['W'],m['H'];els=[]
    for e in m['els']:
        if e.get('opacity',1)<.05:continue
        r=[e['x']/W,e['y']/H,e['w']/W,e['h']/H]
        if e['kind']!='TextElement' and r[2]*r[3]>.9:continue
        els.append(dict(text=normalized_text(e.get('text','')) if e['kind']=='TextElement' else '',r=[round(v,4) for v in r],fs=round(e.get('fs',0)/max(W,H),4),a=round(e['w']/max(.1,e['h']),3)))
    return els

def main():
    excluded={name.rsplit('_',1)[0] for name in json.loads((WIDE/'tournament_set.json').read_text())['tasks']}
    # Also exclude the old audit gallery family outside the tournament if present.
    rows=[];families=set();sourcehash=hashlib.sha256()
    for line in (WIDE/'families.jsonl').open():
        f=json.loads(line)
        if f['fam'] in excluded:continue
        members=f['members']
        pairs=sorted(((a,b) for a in members for b in members if a is not b),key=lambda ab:-abs(math.log(ab[0]['ar']/ab[1]['ar'])))[:3]
        for a,b in pairs:
            ae,be=compact(a),compact(b)
            if len(ae)>16 or len(be)>16 or not ae:continue
            bt={e['text']:e for e in be if e['text']};matched=[]
            for e in ae:
                if e['text'] and e['text'] in bt:matched.append(dict(kind='text',src=e['r'],dst=bt[e['text']]['r'],fs=e['fs']))
            arts=[e for e in ae if not e['text']];other=[e for e in be if not e['text']]
            if len(arts)==len(other):
                arts.sort(key=lambda e:-e['r'][2]*e['r'][3]);other.sort(key=lambda e:-e['r'][2]*e['r'][3])
                matched.extend(dict(kind='art',src=e['r'],dst=t['r'],fs=0) for e,t in zip(arts,other))
            if not any(e['kind']=='text' for e in matched):continue
            matched.sort(key=lambda e:(e['kind'],-e['fs'],-e['src'][2]*e['src'][3]))
            rows.append(dict(source_ar=round(a['ar'],4),target_ar=round(b['ar'],4),items=matched));families.add(f['fam']);sourcehash.update(line.encode())
    out=Path('engine/data');out.mkdir(exist_ok=True)
    (out/'designer_prior.json').write_text(json.dumps(rows,separators=(',',':')))
    (out/'provenance.json').write_text(json.dumps(dict(dataset='CyberAgent Crello multi-size families',license='CDLA-Permissive-2.0',families=len(families),transitions=len(rows),excluded_families=len(excluded),excluded_family_hash=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),input_digest=sourcehash.hexdigest(),contains_pixels=False),indent=2))
    print({'families':len(families),'transitions':len(rows),'bytes':(out/'designer_prior.json').stat().st_size})
if __name__=='__main__':main()
