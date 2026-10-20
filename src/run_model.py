import os, sys, pickle, importlib, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluate as ev, err_public
sys.stdout.reconfigure(encoding="utf-8")
name = sys.argv[1]; params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
mod = importlib.import_module(name)
f = pickle.load(open(os.path.join(os.path.dirname(HERE), "cache", "features_tactical.pkl"), "rb"))
lab = mod.predict(f, params)
print("public :", ev.fmt(ev.score(lab)))
print("full   :", ev.fmt(ev.diagnostics(lab)))
if "--show" in sys.argv: err_public.show(lab)
