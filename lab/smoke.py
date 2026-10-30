import json,time
from pathlib import Path
from engine.resize import diagnose
from engine.objective import violations
from renderer.converter.convert import render_design_to_png
Path('results/smoke').mkdir(parents=True,exist_ok=True)
for path in sorted(Path('input').glob('*.json')):
    task=json.loads(path.read_text());src=task['source'];t=task['target_canvas'];start=time.monotonic()
    winner,info=diagnose(src,t)
    print(path.name,winner['name'],round(time.monotonic()-start,2),'seconds',info['generated'],'candidates',violations(src,winner['output']),flush=True)
    (Path('results/smoke')/(path.stem+'.json')).write_text(json.dumps(winner['output'],indent=2))
    render_design_to_png(winner['output'],'brand_kit/assets',Path('results/smoke')/(path.stem+'.png'),max_size=1200)
