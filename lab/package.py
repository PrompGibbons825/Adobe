"""Package source and evidence, excluding other projects and shared data."""
import json,hashlib,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=[p for base in ('engine','renderer','brand_kit','lab','tests','scripts','input') for p in (root/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [root/p for p in ('README.md','requirements.txt','Dockerfile','chameleon_design_schema_v0.9.json','RESULTS.txt','AGENTS.md','CLAUDE.md','RULES.md','FAQ.md')]
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(root/'results').mkdir(exist_ok=True);(root/'results/source-manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'results/origin-engine.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(root/'results/origin-engine.tar.gz')
