"""Verify the frozen weights and policy settings without running inference."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
if __name__=='__main__':
    manifest=json.loads((ROOT/'models/model-manifest.json').read_text(encoding='utf-8'))
    path=ROOT/manifest['local_file']
    assert path.is_file(),'Download the selected artifact and run scripts/import_model.py --source <path>.'
    assert digest(path)==manifest['sha256'],'Checkpoint checksum mismatch.'
    assert digest(ROOT/'config/policy-v1.json')==manifest['policy_config_sha256'],'Policy settings changed.'
    print('Verified:',manifest['model_version'],manifest['sha256'])
