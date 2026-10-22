# GPU jobs on Google Colab (google-colab-cli)

GPU work runs on Colab so the local machine stays free. The official CLI
(github.com/googlecolab/google-colab-cli) supports Linux/macOS only, so on this Windows machine it runs inside
WSL (Ubuntu); the OAuth login is done once by the user (`colab` prints a URL, the code is pasted back).

```bash
# from Windows: wsl.exe -d Ubuntu-24.04 -- bash -lc "<command>"
colab new -s mtah --gpu T4 --high-mem          # 8 vCPU, ~50 GB RAM, T4
colab upload -s mtah mtah_code.zip /content/mtah_code.zip    # zip of src/*.py, colab/*.py, colab/*.sh
colab exec -s mtah -f colab/launch.py          # unpack + start colab/<JOB>.sh detached, log in /content/mtah/logs
colab exec -s mtah -f colab/poll.py            # tail of the job log, running / stopped
colab download -s mtah /content/mtah/cache/gru/gru_all_dfl.pt ./cache/gru/   # small result files only
colab stop -s mtah                             # always stop the VM when the job ends
```

- Open data (DFL/IDSSE, Metrica sample, SkillCorner, the official scorer) is fetched on the VM by
  `fetch_open_data.py`, so nothing large crosses the local connection.
- No competition (BAR-ATH) data is needed for training / evaluation on open data. Colab is used as compute
  only; Colab's Gemini features must never be pointed at competition data (T&C).
- Jobs print aggregate statistics only.

| job | what | script |
|---|---|---|
| `run_gru` | BiGRU team model vs v7's LightGBM (idea N25), true dead balls, B2 decoder | `src/seq_gru.py`, `src/exp_gru.py` |
| `run_bas` | 2023 SoccerNet ball-action-spotting winner (MIT, one fold) on the BAR-ATH broadcast video: PASS / DRIVE scores (`/content/work/bas_raw_fold<k>.npz`, ~0.4 MB), evaluated locally by `src/bas_eval.py` | `colab/bas_pilot.py` |

`run_bas` needs the match video on the VM without using the local connection: add a Drive shortcut to the shared
match folder ("Add shortcut to My Drive"), then `colab drivemount -s <session>` (the user opens the printed Google
URL, grants all permissions and presses Enter within ~2 minutes; every new VM asks again) and write the video's
path under `/content/drive/.shortcut-targets-by-id/...` to `/content/bas_video.txt` before `launch.py`.
