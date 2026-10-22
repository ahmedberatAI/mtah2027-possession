"""Run on the Colab kernel (colab exec -f colab/poll.py): last lines of a job log + whether it is still running.
The job scripts print aggregates only."""
import os
import subprocess

JOB = "run_gru"
path = "/content/mtah/logs/%s.log" % JOB
if os.path.isfile(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    print("\n".join(lines[-40:]))
# [c] keeps pgrep from matching the shell that runs this very command line
running = subprocess.run("pgrep -f '[c]olab/%s.sh' >/dev/null && echo running || echo stopped" % JOB, shell=True,
                         capture_output=True, text=True).stdout.strip()
print("[job %s: %s]" % (JOB, running))
