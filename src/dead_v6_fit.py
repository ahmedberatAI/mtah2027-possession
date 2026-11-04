"""Re-derive v20's dead-ball models from open data on sim.V6's masks (rounds 46 and 49, docs/RAPOR.md). BAR-ATH's only part is the aggregate shares inside sim.V6 (whole match, no labels, no public window).

    python -I src/dead_v6_fit.py rules       models/dead_rules_open_v2.json: the thresholds and bounds of v1, the weights refitted on V6
    python -I src/dead_v6_fit.py multi       models/dead_multi_v2.txt: the multi-source LightGBM on V6
    TD_V6=1 TD_NORM=1 TD_OUT=tdead6n python -I src/exp_tdead.py train 20261104 (and 20261105, 20261106), then
    python -I src/dead_v6_fit.py td          models/tdead_v1/: the three all-data sequence models
"""
import json
import os
import shutil
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import dead_rules_fit as drf  # noqa: E402
import model_v19  # noqa: E402
import model_v20  # noqa: E402

SOURCES = {"dfl": ("dfl@v6", "dfl", "native"), "metrica": ("metrica@v6g", "metrica@v3", "ctrl"), "sc": ("sc@v6", "sc@v3", "evctrl")}


def rules():
    drf.SOURCES = {"dfl": ("dfl@v6", "dfl@v3", "evctrl"), "metrica": ("metrica@v6g", "metrica@v3", "ctrl"), "sc": ("sc@v6", "sc@v3", "evctrl")}
    data = drf.load()
    with open(model_v19.RULES) as fh:
        v1 = json.load(fh)
    conf = dict(thresholds=v1["thresholds"], bounds=v1["bounds"])
    X, y, w = drf.design(data, conf)
    beta, _ = drf.logistic(X, y, w)
    Xv, yv, off = [], [], []
    for f, g, v, l in data["sc"]:
        m = g[::drf.STRIDE] < 3
        Xv.append(np.stack([(~v)[::drf.STRIDE][m], l[::drf.STRIDE][m]], 1).astype(np.float64))
        yv.append((g[::drf.STRIDE][m] < 2).astype(float))
        off.append((model_v19.rule_terms(f, conf) @ beta)[::drf.STRIDE][m])
    gamma, _ = drf.logistic(np.concatenate(Xv), np.concatenate(yv), np.ones(sum(len(a) for a in yv)), offset=np.concatenate(off))
    conf["beta"], conf["gamma"] = beta.tolist(), gamma.tolist()
    conf["note"] = "Open data only: thresholds and bounds of dead_rules_open_v1.json, weights on sim.V6's masks (src/dead_v6_fit.py rules)."
    with open(model_v20.RULES, "w", newline="\n") as fh:
        json.dump(conf, fh, indent=1)
    print("beta", np.round(beta, 3).tolist(), "gamma", np.round(gamma, 3).tolist())


def multi():
    model = model_v19.fit_dead({k: (bench.features(model_v19.VERSION, v[0]), bench.labels(v[1], v[2])) for k, v in SOURCES.items()})
    model.save_model(model_v20.M_MODEL)
    print("written", model_v20.M_MODEL)


def td():
    os.makedirs(model_v20.TD_DIR, exist_ok=True)
    for seed in model_v20.TD_SEEDS:
        shutil.copyfile(os.path.join(bench.CACHE, "tdead6n", "all_%d.pt" % seed), os.path.join(model_v20.TD_DIR, "tdead_v6n_s%d.pt" % seed))
    print("copied", model_v20.TD_DIR)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    {"rules": rules, "multi": multi, "td": td}[sys.argv[1]]()
