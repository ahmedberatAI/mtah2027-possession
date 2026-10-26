"""Aggregate Gradient/PFF schema audit; never publishes raw match records.

Events v2.5 and tracking v2.2 have DIFFERENT field naming/version conventions.
This validates inventory, real event coverage, ambiguity flags and one match's
tracking timing/duplication. It does not assert equivalence to competition GT.
"""
import argparse
import bz2
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_data/gradient'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-match', default='10511')
    args = parser.parse_args()
    manifest = json.loads((DATA / 'drive_manifest.json').read_text(encoding='utf-8'))
    inventory = {}
    for kind, files in manifest.items():
        expected = [f for f in files if f['title'].split('.')[0].isdigit()]
        complete = [f for f in expected if (DATA/kind/f['title']).exists() and
                    (DATA/kind/f['title']).stat().st_size == int(f['size'])]
        inventory[kind] = dict(expected=len(expected), complete=len(complete),
                               expected_bytes=sum(int(f['size']) for f in expected))
    counters = {k: Counter() for k in ('event_type', 'challenge_type', 'challenge_outcome',
                 'touch_type', 'rebound_outcome', 'initial_touch_type', 'period')}
    totals = Counter()
    per_match = {}
    for f in sorted((DATA/'events').glob('*.json')):
        rows = json.loads(f.read_text(encoding='utf-8'))
        unique_game = set()
        own = {}
        conflict_seq = set()
        for r in rows:
            g, p, it = (r.get('gameEvents') or {}), (r.get('possessionEvents') or {}), (r.get('initialTouch') or {})
            unique_game.add(r['gameEventId'])
            for key, value in [('event_type',p.get('possessionEventType')),
                               ('challenge_type',p.get('challengeType')),('challenge_outcome',p.get('challengeOutcomeType')),
                               ('touch_type',p.get('touchType')),('rebound_outcome',p.get('reboundOutcomeType')),
                               ('initial_touch_type',it.get('initialTouchType')),('period',g.get('period'))]:
                if value is not None:
                    counters[key][str(value)] += 1
            totals['rows'] += 1
            totals['initial_non_event'] += g.get('initialNonEvent') is True
            totals['event_non_event'] += p.get('nonEvent') is True
            totals['video_missing'] += g.get('videoMissing') is True
            totals['deflector_nonnull'] += p.get('deflectorPlayerId') is not None
            totals['missing_home_team'] += not isinstance(g.get('homeTeam'), bool)
            seq = r.get('sequence')
            if seq is not None and isinstance(g.get('homeTeam'), bool) and g.get('gameEventType') == 'OTB':
                seq_key = (g.get('period'), seq)
                own.setdefault(seq_key, set()).add(g['homeTeam'])
                if len(own[seq_key]) > 1:
                    conflict_seq.add(seq_key)
        per_match[f.stem] = dict(rows=len(rows), unique_game_events=len(unique_game),
                                 sequences=len(own), sequences_with_multiple_actor_teams=len(conflict_seq))
        print('event audit',f.stem,len(rows),'rows',flush=True)
    tracking = None
    f = DATA/'tracking'/f'{args.tracking_match}.jsonl.bz2'
    if f.exists():
        agg = Counter()
        ge_types, pe_types, periods = Counter(), Counter(), Counter()
        times = {}
        conflicting = set()
        labels = {}
        with bz2.open(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                frame = r['frameNum']
                ge, pe = r.get('game_event') or {}, r.get('possession_event') or {}
                agg['rows'] += 1
                times[frame] = r['videoTimeMs']
                ge_types[str(ge.get('game_event_type'))] += 1
                pe_types[str(pe.get('possession_event_type'))] += 1
                periods[str(r.get('period'))] += 1
                label = ge.get('home_ball')
                if frame in labels and labels[frame] != label:
                    conflicting.add(frame)
                labels[frame] = label
                agg['home_ball_null_rows'] += not isinstance(label, bool)
                agg['actor_differs_from_sequence_team'] += isinstance(label, bool) and isinstance(ge.get('home_team'), bool) and label != ge['home_team']
                agg['ball_visible_rows'] += any(b.get('visibility') == 'VISIBLE' for b in (r.get('balls') or []))
        ff = sorted(times)
        gaps = Counter(ff[i]-ff[i-1] for i in range(1,len(ff)))
        dt = Counter(round((times[ff[i]]-times[ff[i-1]])/(ff[i]-ff[i-1]),4) for i in range(1,len(ff)))
        tracking = dict(match=args.tracking_match, counts=dict(agg), unique_frames=len(ff),
                        duplicate_rows=agg['rows']-len(ff), ownership_conflict_frames=len(conflicting),
                        frame_step_histogram=dict(gaps), milliseconds_per_frame=dict(dt.most_common(5)),
                        game_event_types=dict(ge_types),possession_event_types=dict(pe_types),periods=dict(periods))
    result = dict(inventory=inventory, event_totals=dict(totals),
                  event_counts={k:dict(v) for k,v in counters.items()},per_match=per_match,tracking_sample=tracking,
                  status='schema_audit_only_not_training_labels')
    out = ROOT/'reports/gates/gradient_audit.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_match'},indent=2),flush=True)


if __name__ == '__main__':
    main()
