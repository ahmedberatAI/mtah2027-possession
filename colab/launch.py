"""Run on the Colab kernel (colab exec -f colab/launch.py): unpack the uploaded code and start a job script in
the background, detached from the kernel, logging to /content/mtah/logs/<job>.log. JOB is replaced by the caller."""
import os
import subprocess
import zipfile

JOB = "run_gru"
ROOT = "/content/mtah"
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile("/content/mtah_code.zip") as z:
    z.extractall(ROOT)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
log = open(os.path.join(ROOT, "logs", JOB + ".log"), "ab")
p = subprocess.Popen(["bash", os.path.join(ROOT, "colab", JOB + ".sh")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
print("started", JOB, "pid", p.pid)
