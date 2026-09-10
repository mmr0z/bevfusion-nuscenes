#!/usr/bin/env python3
"""Download private release assets with gh and verify their SHA-256 hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'artifacts/checkpoints.json').read_text())
    for item in manifest['files']:
        target = ROOT / item['path']
        if target.exists():
            if digest(target) == item['sha256']:
                print(f'OK: {item["path"]}')
                continue
            raise RuntimeError(f'Istniejący plik ma inną sumę SHA-256: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            subprocess.run(['gh', 'release', 'download', manifest['release'], '--repo',
                            manifest['repository'], '--pattern', item['asset'], '--dir', temp], check=True)
            source = Path(temp) / item['asset']
            if source.stat().st_size != item['bytes'] or digest(source) != item['sha256']:
                raise RuntimeError(f'Niepoprawny plik: {item["asset"]}')
            source.rename(target)
        print(f'Pobrano: {item["path"]}')


if __name__ == '__main__':
    main()
