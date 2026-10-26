"""Download only small SoccerTrack v1.2 BAS annotations; audit schema/counts.

Do not use the auxiliary ball trajectories as independent validation: they
are reconstructed from these same events (explicit dataset-card warning).
"""
from collections import Counter
import json
import os
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'external_data/soccertrack_v2'
REPO = 'atomscott/soccertrack-v2'
TAG = 'v1.2'


def main():
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504))))
    meta = session.get(f'https://huggingface.co/api/datasets/{REPO}/revision/{TAG}', timeout=60)
    meta.raise_for_status()
    sha = meta.json()['sha']
    if os.environ.get('HF_TOKEN'):
        session.headers['Authorization'] = 'Bearer ' + os.environ['HF_TOKEN']
    r = session.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{sha}?recursive=true&limit=1000', timeout=60)
    r.raise_for_status()
    files = [f for f in r.json() if f['path'] == 'README.md' or (f['path'].startswith('bas/') and f['path'].endswith('.json'))]
    counts, per_match, missing, keys = Counter(), {}, Counter(), Counter()
    for f in files:
        path = OUT / f['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size != f['size']:
            data = session.get(f'https://huggingface.co/datasets/{REPO}/resolve/{sha}/{f["path"]}', timeout=60)
            if data.status_code in (401, 403):
                report = ROOT / 'reports/gates/soccertrack_audit.json'
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps(dict(repository=REPO, tag=TAG, revision=sha,
                    status='access_required', gated=meta.json().get('gated'),
                    metadata_files=len(files),
                    license='CC-BY-4.0 (public dataset card)',
                    caveat='Annotations require authorized HF access. No event content audited.'), indent=2), encoding='utf-8')
                print('SoccerTrack: CC-BY-4.0; Hugging Face auto-gated, authorized access required.')
                return
            data.raise_for_status()
            path.write_bytes(data.content)
        if path.suffix == '.json':
            evs = json.loads(path.read_text())['annotations']
            per_match[path.parent.name] = len(evs)
            for e in evs:
                counts[e['label']] += 1
                keys.update(e.keys())
                for k in ('frame', 'position', 'team', 'player_id'):
                    missing[k] += int(e.get(k) is None)
    summary = dict(repository=REPO, tag=TAG, revision=sha, matches=per_match,
                   total=sum(counts.values()), labels=dict(counts), missing=dict(missing), schema_keys=dict(keys),
                   license='CC-BY-4.0 (dataset card)',
                   caveat='Auxiliary ball trajectories are event-anchored interpolation; not independent tracking GT.')
    report = ROOT / 'reports/gates/soccertrack_audit.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
