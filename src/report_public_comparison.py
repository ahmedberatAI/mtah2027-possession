"""Summarize official aggregate metrics; never print frame-level match content."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'cache/github_public_comparison/official'
OUT=ROOT/'reports/github_public_comparison'


def main():
    rows=list(csv.DictReader((SOURCE/'scoreboard.csv').open()))
    assert len(rows)==12 and all(r['status']=='scored' for r in rows)
    fields=('submission','status','spell_f1_mean','macro_f1','trans_f1_25','poss','dur',
            'mean_abs_latency','spell_bootstrap_low','spell_bootstrap_high')
    out=[{k:r[k] for k in fields} for r in rows]
    (OUT/'metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    ensemble=json.loads((SOURCE/'reports/bigru_ensemble_equal.json').read_text())
    keys=('spell_f1','n_gt_spells','n_pred_spells','gt_mean_spell_duration_s',
          'mean_spell_duration_s','n_graded','n_abstain_on_graded','alive_dead_f1','team_acc_both_alive')
    (OUT/'ensemble_diagnostics.json').write_text(json.dumps({k:ensemble[k] for k in keys},indent=2),encoding='utf-8')
    lines=['# Aynı açık bölümde resmî skor karşılaştırması','',
           '| Model | spellF1 | macroF1 | transF1 ±1 sn | Possession | Ort. süre (sn) | spellF1 bootstrap %95 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        metrics=' | '.join(f'{float(r[k]):.4f}' for k in ('spell_f1_mean','macro_f1','trans_f1_25'))
        lines.append(f'| {r["submission"]} | {metrics} | {r["poss"]} | {float(r["dur"]):.1f} | '
                     f'[{float(r["spell_bootstrap_low"]):.4f}, {float(r["spell_bootstrap_high"]):.4f}] |')
    (OUT/'METRICS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified 12 scored submissions; exported aggregate metrics only.')


if __name__=='__main__':main()
