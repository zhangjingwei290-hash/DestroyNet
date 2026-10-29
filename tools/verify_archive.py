"""Verify packaged files, Python syntax and recorded numerical tables; no models loaded."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
 for p,d in manifest.items():assert sha(ROOT/p)==d,p
 copies=json.loads((ROOT/'ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))['files']
 for row in copies:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 sources=list(ROOT.rglob('*.py'))
 for p in sources:ast.parse(p.read_text(encoding='utf-8-sig'))
 s=json.loads((ROOT/'versions/v5/results/summary.json').read_text(encoding='utf-8'))
 assert (s['raw_runs'],s['paired_blocks'],s['instances'])==(1800,900,45)
 assert sha(ROOT/'versions/v5/checkpoints/meta_complete.pt')=='1a3493e34e4c23f7eae7c465ba6c48a69d49884a01ec80d599eec6cd9c48187f'
 for version in ['v4','v4_3']:
  s=json.loads((ROOT/f'versions/{version}/results/summary.json').read_text())
  assert len(s['comparison'])==45 and len(s['search_seeds'])==5
  assert len(list((ROOT/f'versions/{version}/checkpoints').glob('direction_*.pt')))==16
 assert len(list((ROOT/'data/v5/instances').glob('*.json')))==270
 print(json.dumps(dict(passed=True,manifest_files=len(manifest),original_byte_copies=len(copies),python_files_parsed=len(sources),neural_model_loaded=False,search_or_training_executed=False)))
if __name__=='__main__':main()
