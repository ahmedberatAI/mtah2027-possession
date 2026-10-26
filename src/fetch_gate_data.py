"""Fetch pinned SkillCorner files needed by the gates, including dynamic events."""
import concurrent.futures
import importlib.util
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fetch_open', ROOT / 'colab/fetch_open_data.py')
fetch_open = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_open)


def job(m):
    for suffix in ('match.json', 'phases_of_play.csv', 'dynamic_events.csv', 'tracking_extrapolated.jsonl'):
        host = 'media.githubusercontent.com/media' if suffix.endswith('jsonl') else 'raw.githubusercontent.com'
        name = f'{m}_{suffix}'
        url = f'https://{host}/SkillCorner/opendata/{fetch_open.SC_SHA}/data/matches/{m}/{name}'
        path = ROOT / 'external_data/skillcorner/data/matches' / m / name
        for attempt in range(4):
            try:
                fetch_open.fetch(url, str(path))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    print(f'SkillCorner {m} complete', flush=True)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, fetch_open.SC_MATCHES))
