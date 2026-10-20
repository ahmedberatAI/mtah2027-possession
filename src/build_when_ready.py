"""Parse + degrade + featurise each DFL match as soon as its positions file is complete."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dfl, train_dfl
sys.stdout.reconfigure(encoding="utf-8")
sizes = {}
for line in open(os.path.join(dfl.IDSSE, "urls.txt")):
    url, name, size = line.split()
    sizes[name] = int(size)
pending = list(dfl.MATCHES)
while pending:
    for m in list(pending):
        try:
            path = dfl._file("positions_raw_observed", m)
        except IOError:
            continue
        if os.path.getsize(path) != sizes[os.path.basename(path)]:
            continue
        t = time.time()
        dfl.parse(m)
        train_dfl.build()   # builds every match whose npz exists and feature pickle is missing
        print("ready", m, "%.0fs" % (time.time() - t), flush=True)
        pending.remove(m)
    time.sleep(10)
print("ALL BUILT")
