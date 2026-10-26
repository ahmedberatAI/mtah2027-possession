"""Materialize provider-returned file references into the local data directory.

The input is a PRIVATE cache manifest from the Drive connector, not a public
artifact: signed URLs must not be logged or committed. Every file is checked
against the provider's expected length before atomically replacing its target.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]


def fetch(f):
    dest = ROOT / 'external_data/gradient' / f['kind'] / f['name']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == f['expected_bytes']:
        return
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1)))
    part = dest.with_name(dest.name + '.part')
    try:
        with session.get(f['download_url'], stream=True, timeout=(30, 120)) as r:
            if r.status_code != 200:
                raise RuntimeError(f"download status {r.status_code}: {f['kind']}/{f['name']}")
            with part.open('wb') as out:
                for block in r.iter_content(1 << 20):
                    out.write(block)
        if part.stat().st_size != f['expected_bytes']:
            raise RuntimeError(f"size mismatch: {f['kind']}/{f['name']}")
        part.replace(dest)
        print(f"saved {f['kind']}/{f['name']} {f['expected_bytes']} bytes", flush=True)
    except requests.RequestException as exc:
        # Exception text can contain the signed URL; never print it.
        raise RuntimeError(f"{type(exc).__name__}: {f['kind']}/{f['name']}") from None


if __name__ == '__main__':
    entries = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, entries))
