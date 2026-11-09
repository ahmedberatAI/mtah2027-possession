"""SoccerTrack's emissions from the existing all-data stackers (no fold nets are needed there: every model is unseen on it). Open data only."""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import warnings
    warnings.filterwarnings("ignore")
    os.chdir(REPO)
    import numpy as np
    import torch
    import exp_stack as xs
    from exp_stack import OUT, bench, ec
    from foldnets import OUTD, emit_half
    os.makedirs(OUTD, exist_ok=True)
    torch.set_num_threads(6)
    for s in (0, 1, 2):
        d = bench._load(os.path.join(OUT, "inputs_st_%d.pkl" % s))
        ck = torch.load(os.path.join(OUT, "sall_Hy_%d.pt" % s), map_location="cpu", weights_only=False)
        net = xs.make_net("Hy", ck["d_in"], ck.get("e", 0))
        net.load_state_dict(ck["state"])
        net.eval()
        out = []
        for r, (q_, a_) in zip(d["rows"], xs.bf_mean("st@v3", (0, 1, 2))):
            z = ec.pool(xs._bf_cols(q_, a_).astype(np.float64)).astype(np.float32)
            n_ = min(len(r[1]), len(z))
            out.append(emit_half(net, np.hstack([r[1][:n_], z[:n_]])))
        with open(os.path.join(OUTD, "emit_Hy_%d_st.pkl" % s), "wb") as fh:
            pickle.dump(out, fh)
        print("seed %d done; t = %s" % (s, np.round(net.t.detach().numpy(), 2).tolist()), flush=True)


if __name__ == "__main__":
    main()
