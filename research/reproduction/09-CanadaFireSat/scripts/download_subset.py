"""Bounded, resumable public-data download; validates server sizes and LFS SHA256."""
import concurrent.futures as futures
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
DATASET = 'EPFL-ECEO/CanadaFireSat'
MODEL = 'EPFL-ECEO/canadafiresat-multi-resnet-50'

def api(path):
    response = requests.get(ENDPOINT + path, timeout=(20, 60))
    response.raise_for_status()
    return response.json()

def download(item):
    path = ROOT / item['local']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('size')
    if path.exists() and expected and path.stat().st_size == expected:
        return {'path': item['local'], 'bytes': expected, 'cached': True}
    temp = path.with_suffix(path.suffix + '.part')
    started = time.time()
    for attempt in range(8):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            response = requests.get(item['url'], stream=True, timeout=(30, 120),
                                    headers={'Range': f'bytes={offset}-'} if offset else {})
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            with temp.open('ab' if append else 'wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            if expected and temp.stat().st_size != expected:
                raise RuntimeError(f'Size mismatch: {temp.stat().st_size} != {expected}')
            if item.get('sha256'):
                digest = hashlib.sha256()
                with temp.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != item['sha256']:
                    raise RuntimeError('LFS SHA256 mismatch')
            temp.replace(path)
            result = {'path': item['local'], 'bytes': path.stat().st_size,
                      'seconds': round(time.time() - started, 2)}
            print(json.dumps(result), flush=True)
            return result
        except Exception as exc:
            print(f'RETRY {attempt + 1} {item["local"]}: {exc}', flush=True)
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)

def main():
    record = json.loads((ROOT / 'configs/download_manifest.json').read_text(encoding='utf-8-sig'))
    manifest = record['files']
    assert sum(item['size'] for item in manifest) == record['total_bytes']
    assert record['total_bytes'] < 12 * 10**9
    for item in manifest:
        relative = Path(item['local'])
        assert not relative.is_absolute() and '..' not in relative.parts
        if ENDPOINT != 'https://huggingface.co':
            item['url'] = item['url'].replace('https://huggingface.co', ENDPOINT, 1)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    print(f"Fixed download: {len(manifest)} files, {record['total_bytes']/1e9:.3f} GB", flush=True)
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, manifest))
    (ROOT / 'outputs/download_complete.json').write_text(json.dumps(results, indent=2))
    print('DOWNLOAD_COMPLETE', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (ROOT / 'outputs/download_failed.json').write_text(json.dumps({'error': str(exc)}))
        raise
