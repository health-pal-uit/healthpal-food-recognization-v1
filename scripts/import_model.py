"""Copy a downloaded checkpoint into this service after verifying its identity."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);args=parser.parse_args()
    manifest=json.loads((ROOT/'models/model-manifest.json').read_text(encoding='utf-8'))
    if not args.source.is_file() or digest(args.source)!=manifest['sha256']:
        raise SystemExit('The source file is not the frozen selected checkpoint.')
    target=ROOT/manifest['local_file'];target.parent.mkdir(exist_ok=True)
    if target.exists():
        if digest(target)!=manifest['sha256']:raise SystemExit('Existing model differs; preserve it and inspect before replacing.')
    else:shutil.copyfile(args.source,target)
    print('Selected checkpoint verified:',target)
