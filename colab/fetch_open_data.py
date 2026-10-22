"""Fetch the OPEN datasets the bench needs, on the remote (Colab) VM -- the VM's network, not the local one.

  open DFL / IDSSE      figshare 28196177 (CC BY 4.0), via src/download_idsse.py
  Metrica sample games  github.com/metrica-sports/sample-data at a pinned commit, via metrica_sample.download()
  SkillCorner open data github.com/SkillCorner/opendata at a pinned commit (tracking files are Git LFS)
  official scorer       github.com/metrica-sports/more-than-a-hack-2027 (possession_metrics.py, public)

No competition (BAR-ATH) data is involved.

  python colab/fetch_open_data.py <repo root>
"""
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
SC_SHA = "4340d274572876239c154c90bc507a9b3250a656"
SC_MATCHES = ["1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691", "1996435",
              "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166", "2013725", "2015213",
              "2016236", "2017461"]
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(path + ".part", "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(path + ".part", path)


def main():
    off = os.path.join(ROOT, "external", "more-than-a-hack-2027")
    if not os.path.isdir(off):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/metrica-sports/more-than-a-hack-2027", off],
                       check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "download_idsse.py")], check=True)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import metrica_sample
    metrica_sample.download()
    base = os.path.join(ROOT, "external_data", "skillcorner", "data", "matches")
    for m in SC_MATCHES:
        d = os.path.join(base, m)
        for name, host in (("%s_match.json" % m, "raw.githubusercontent.com/SkillCorner/opendata"),
                           ("%s_phases_of_play.csv" % m, "raw.githubusercontent.com/SkillCorner/opendata"),
                           ("%s_tracking_extrapolated.jsonl" % m, "media.githubusercontent.com/media/SkillCorner/opendata")):
            fetch("https://%s/%s/data/matches/%s/%s" % (host, SC_SHA, m, name), os.path.join(d, name))
        print("skillcorner", m, "ok", flush=True)
    print("open data ready", flush=True)


if __name__ == "__main__":
    main()
