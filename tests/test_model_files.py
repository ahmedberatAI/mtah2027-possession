"""The shipped LightGBM models load the same from a CRLF checkout (Windows, core.autocrlf=true), and git keeps them
byte for byte. LightGBM 4.7 aborts the whole process when given the path of a CRLF model file."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import bench  # noqa: E402

MODELS = ["models/v10_team.txt", "models/short_verifier_ins_g.txt", "models/short_verifier_del_g.txt",
          "models/short_verifier_ins_g_blend50.txt", "models/short_verifier_del_g_blend50.txt",
          "models/short_verifier_ins_c1m25s.txt", "models/short_verifier_del_c1m25s.txt",
          "models/short_verifier_ins_j1m25s.txt", "models/short_verifier_del_j1m25s.txt", "models/dead_base_v3.txt",
          "models/dead_multi_v1.txt"]


class ModelFileTests(unittest.TestCase):
    def test_crlf_copy_predicts_the_same(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            for rel in MODELS:
                lf = open(os.path.join(ROOT, rel), "rb").read().replace(b"\r\n", b"\n")    # as git stores it
                path = os.path.join(d, os.path.basename(rel))
                with open(path, "wb") as fh:
                    fh.write(lf.replace(b"\n", b"\r\n"))                                   # as autocrlf checks out
                a, b = bench.load_booster(os.path.join(ROOT, rel)), bench.load_booster(path)
                X = rng.normal(size=(500, a.num_feature()))
                X[::5, ::3] = np.nan
                self.assertTrue(np.array_equal(a.predict(X), b.predict(X)), rel)

    def test_git_keeps_models_unconverted(self):
        out = subprocess.run(["git", "check-attr", "text", "--"] + MODELS, cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("not a git checkout")
        for line in out.stdout.splitlines():
            self.assertTrue(line.endswith(": text: unset"), line)


if __name__ == "__main__":
    unittest.main()
