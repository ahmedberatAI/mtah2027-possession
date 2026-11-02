"""PFF raw broadcast tracking -> 25 Hz arrays + conservative possession proxies.

Never use *Smoothed positions: the provider constructs smoothed BALL from labels.
Labels are event-derived proxies, not a claim of equivalence to private BAR-ATH GT.
Time alignment is validated using shared possession-event IDs, per available half.
Unknown / invalid / video-missing intervals and tracking cuts are training masks X.
"""
from collections import Counter
from pathlib import Path
import bz2
import json
import numpy as np
import orjson
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_data/gradient'
CACHE = ROOT / 'cache/pff25'
VERSION = 'raw_control_v2'
FPS = 25
H, A, D, X = 0, 1, 2, 3
RELEASE = {'PA', 'CR', 'SH', 'CL'}
DIRECT = RELEASE | {'BC', 'IT'}
DUELS = {'A', 'FIFTY', 'G', 'K'}
MAX_GAP = .12


def matches():
    return sorted(p.stem for p in (DATA/'events').glob('*.json'))


def read_json(path):
    return orjson.loads(Path(path).read_bytes())


def metadata(match):
    m = read_json(DATA/'metadata'/f'{match}.json')
    return m[0] if isinstance(m, list) else m


def groups_for(rows, per):
    out = {}
    for r in rows:
        if int((r.get('gameEvents') or {}).get('period') or 0) == per:
            out.setdefault(r['gameEventId'], []).append(r)
    return sorted(out.values(), key=lambda rs: (rs[0]['startTime'], rs[0]['gameEventId']))


def control_time(rs):
    """Earliest supported control. Duels/rebounds/failed touches do not switch team.

    IT explicitly represents first control (its bad/miscontrol quality still
    retains possession per spec p7). Non-duel challenges begin with a carrier
    controlling the ball. A duel alone is not enough, even if it is won.
    """
    start = float(rs[0]['startTime'])
    first = min(float(r['eventTime']) for r in rs)
    times = []
    for r in rs:
        p = r.get('possessionEvents') or {}
        typ = p.get('possessionEventType')
        if p.get('nonEvent'):
            continue
        t = float(r['eventTime'])
        if typ == 'IT':
            times.append(t)
        elif typ in DIRECT or (typ == 'TC' and p.get('touchType') in {'H','T'}):
            # A one-touch action at the initial contact is control at that contact.
            # Do not backdate a later pass to a preceding loose-ball duel.
            times.append(start if abs(t-first) <= .04 else t)
        elif typ == 'CH' and p.get('challengeType') not in DUELS and p.get('challengeType') is not None:
            times.append(start)
    return min(times) if times else None


def restart_time(rs, ct):
    g = rs[0]['gameEvents']
    if g.get('setpieceType') not in (None, 'O'):
        kick = [float(r['eventTime']) for r in rs if
                (r.get('possessionEvents') or {}).get('possessionEventType') in RELEASE and
                not (r.get('possessionEvents') or {}).get('nonEvent')]
        return min(kick) if kick else None
    return ct


def _owners(grid, anchors):
    if not anchors:
        return np.full(len(grid), X, np.int8), np.full(len(grid), -1, np.int64)
    anchors = sorted(anchors, key=lambda x: (x[0], x[1] == X))
    # Conflicting simultaneous owners become unknown rather than arbitrary order.
    merged = []
    for t, team, end in anchors:
        if merged and abs(t-merged[-1][0]) < 1e-7:
            old = merged[-1]
            merged[-1] = (t, team if old[1] == team else X, max(end, old[2]))
        else:
            merged.append((t, team, end))
    at = np.array([a[0] for a in merged])
    tm = np.array([a[1] for a in merged], np.int8)
    expiry = np.array([max(a[0], a[2])+15 for a in merged])
    idx = np.searchsorted(at, grid+1e-9, side='right')-1
    valid = idx >= 0
    out = np.full(len(grid), X, np.int8)
    out[valid] = tm[idx[valid]]
    out[valid & (grid > expiry[np.maximum(idx,0)])] = X
    af = np.searchsorted(grid, at, side='left')
    return out, np.where(valid, af[np.maximum(idx,0)], -1)


def labels_from_events(rows, per, grid, offset=0.):
    """Return native, controlled, and label diagnostics on an absolute video grid."""
    native, controlled, restarts, stops, invalid = [], [], [], [], []
    stats = Counter()
    for rs in groups_for(rows, per):
        r = rs[0]; g = r['gameEvents']; typ = g.get('gameEventType')
        start = float(r['startTime'])+offset
        end = max(float(x.get('endTime') or x['eventTime']) for x in rs)+offset
        if typ in {'OUT','END'}:
            stops.append(float(r['eventTime'])+offset)
        if typ and typ.endswith('KICKOFF'):
            restarts.append(float(r['eventTime'])+offset)
            if isinstance(g.get('homeTeam'), bool):
                owner = H if g['homeTeam'] else A
                native.append((start,owner,end)); controlled.append((start,owner,end))
        if typ != 'OTB':
            continue
        bad = g.get('initialNonEvent') or g.get('videoMissing') or any(
            (x.get('possessionEvents') or {}).get('nonEvent') for x in rs)
        if bad or not isinstance(g.get('homeTeam'),bool):
            native.append((start,X,end)); controlled.append((start,X,end))
            invalid.append((start,end)); stats['invalid_groups'] += 1
            continue
        team = H if g['homeTeam'] else A
        ct = control_time(rs)
        native.append((start,team,end))
        if ct is not None:
            controlled.append((ct+offset,team,end))
            stats['confirmed_control_groups'] += 1
        else:
            stats['noncontrolling_groups_held'] += 1
        rt = restart_time(rs,ct)
        if rt is not None:
            restarts.append(rt+offset)
            if g.get('setpieceType') not in (None,'O'):
                # The kick establishes live ownership even if the initial ball
                # placement/control anchor occurred during the stoppage.
                native.append((rt+offset,team,end)); controlled.append((rt+offset,team,end))
    # A stop remains D until the next confirmed control / set-piece kick.
    # Restarts before a stop do not affect that stop's end.
    dead = np.zeros(len(grid),bool)
    rr = np.sort(restarts)
    if len(rr):
        dead[grid < rr[0]-1e-9] = True
    for t in stops:
        ix = np.searchsorted(rr,t+1e-7)
        end = rr[ix] if ix < len(rr) else np.inf
        dead |= (grid >= t-1e-9) & (grid < end-1e-9)
    n, ni = _owners(grid,native); c, ci = _owners(grid,controlled)
    # No ownership inherited across a stoppage before a new live control anchor.
    run_start = np.maximum.accumulate(np.where(np.r_[True, dead[:-1]] & ~dead, np.arange(len(grid)), 0))
    n[(ni < run_start) & ~dead] = X
    c[(ci < run_start) & ~dead] = X
    n[dead] = D; c[dead] = D
    for a,b in invalid:
        mask=(grid >= a-1e-9)&(grid <= b+1e-9)&~dead
        n[mask]=X;c[mask]=X
    return n,c,dict(stats)


def team_events(rows, per, grid, shift):
    """(grid index, team, possessionEventType, outcome) of every on-the-ball event of one period."""
    out = []
    for rs in groups_for(rows, per):
        g = rs[0]['gameEvents']
        if not isinstance(g.get('homeTeam'), bool):
            continue
        team = H if g['homeTeam'] else A
        for r in rs:
            p = r.get('possessionEvents') or {}
            kind = p.get('possessionEventType')
            if not kind or p.get('nonEvent'):
                continue
            oc = p.get('passOutcomeType') or p.get('crossOutcomeType') or p.get('shotOutcomeType')
            f = int(round((float(r['eventTime'])+shift-grid[0])*FPS))
            if 0 <= f < len(grid):
                out.append((f, team, kind, oc))
    return out


def controlled_reading(lab, evs, keep_s=.5):
    """The challenge's controlled-possession reading (src/metrica_sample.py merge_uncontrolled, the rule behind the
    Metrica 'ctrl' and DFL 'evctrl' labels) on PFF's labels: a spell that directly follows the other team's spell (no
    dead ball between) is a change of possession only if the new team shows control inside it -- a completed pass or
    cross, a shot, or any of its events >= keep_s after the spell starts; otherwise it goes back to the team that had
    the ball. keep_s .5 matches the reference statistics (round 24: 217 possessions per match, 21% < 4 s, 10 in play
    between two opponent possessions; Metrica ctrl 231 / 20% / 12, DFL evctrl 229 / 18% / 13)."""
    out = np.asarray(lab).copy()
    st = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    sp = list(zip(st, np.r_[st[1:], len(out)], out[st]))
    keep = int(keep_s*FPS)
    for i in range(1, len(sp)):
        a, b, k = sp[i]
        pa = sp[i-1][0]
        if sp[i-1][1] != a or k > 1 or out[pa] not in (H, A) or out[pa] == k:
            continue
        inside = [(f, kind, oc) for f, t, kind, oc in evs if a <= f < b and t == k]
        if not (any((kind in ('PA', 'CR') and oc == 'C') or kind == 'SH' for f, kind, oc in inside)
                or any(f >= a+keep for f, kind, oc in inside)):
            out[a:b] = out[pa]
    return out


def ctrl_labels(match, keep_s=.5):
    """{period: controlled-reading labels} on parse()'s grid (the cached npz and its alignment report)."""
    path, info = parse(match)
    shifts = {p['period']: p['applied_shift'] for p in info['periods']}
    rows = read_json(DATA/'events'/f'{match}.json')
    out = {}
    with np.load(path) as d:
        for per in d['periods']:
            lab = d[f'h{per}_labels']
            out[int(per)] = controlled_reading(lab, team_events(rows, int(per), d[f'h{per}_time'], shifts[int(per)]),
                                               keep_s)
    return out


def resample(t, values, grid, max_gap=MAX_GAP):
    """Linear interpolation only within short observed intervals, never through cuts."""
    j = np.clip(np.searchsorted(t,grid,side='right')-1,0,len(t)-2)
    width=t[j+1]-t[j]
    w=(grid-t[j])/width
    shape=(-1,)+(1,)*(values.ndim-1)
    a,b=values[j],values[j+1]
    out=a*(1-w.reshape(shape))+b*w.reshape(shape)
    exact=np.abs(grid-t[j]) < 1e-7
    out[exact]=a[exact]
    valid=(grid>=t[0]-1e-7)&(grid<=t[-1]+1e-7)&((width<=max_gap)|exact)
    out[~valid]=np.nan
    return out.astype(np.float32),valid


def parse(match, refresh=False):
    dest=CACHE/VERSION/f'{match}.npz'; report=dest.with_suffix('.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and report.exists() and not refresh:
        return dest,read_json(report)
    meta=metadata(match)
    pitch=meta['stadium']['pitches'][0]
    L,W=float(pitch['length']),float(pitch['width'])
    roster=read_json(DATA/'rosters'/f'{match}.json')
    home=int(meta['homeTeam']['id'])
    keys=sorted(set((0 if int(p['team']['id'])==home else 1,int(p['shirtNumber'])) for p in roster))
    slot={k:i for i,k in enumerate(keys)}
    team=np.array([k[0] for k in keys],np.int8)
    gks={(0 if int(p['team']['id'])==home else 1,int(p['shirtNumber'])) for p in roster if p['positionGroupType']=='GK'}
    gk=np.array([k in gks for k in keys],bool)
    events=read_json(DATA/'events'/f'{match}.json')
    pe={r['possessionEventId']:r for r in events if r.get('possessionEventId') is not None}
    parts={}; stats=Counter(); residuals={}
    with bz2.open(DATA/'tracking'/f'{match}.jsonl.bz2','rb') as fh:
        for line in fh:
            r=orjson.loads(line); per=int(r['period']); fr=int(r['frameNum']); t=float(r['videoTimeMs'])/1000
            stats['raw_rows']+=1
            if per not in (1,2,3,4): continue
            part=parts.setdefault(per,dict(t=[],P=[],B=[],seen={}))
            pp=r.get('possession_event') or {}; pid=r.get('possession_event_id')
            if pid in pe and pp.get('start_frame')==fr and int(pe[pid]['gameEvents']['period'])==per:
                residuals.setdefault(per,{})[pid]=t-float(pe[pid]['eventTime'])
            p=np.full((len(keys),2),np.nan,np.float32)
            for k,field in enumerate(('homePlayers','awayPlayers')):
                for q in r.get(field) or []:
                    try: key=(k,int(q.get('jerseyNum')))
                    except (ValueError,TypeError): stats['unknown_jersey']+=1;continue
                    if key not in slot or q.get('x') is None or q.get('y') is None: continue
                    j=slot[key];v=np.array([.5+float(q['x'])/L,.5+float(q['y'])/W],np.float32)
                    if np.isfinite(p[j,0]) and np.linalg.norm((p[j]-v)*[L,W])>2:
                        p[j]=np.nan;stats['duplicate_player_conflict']+=1
                    else: p[j]=v
            balls=r.get('balls') or []
            if len(balls)>1:
                balls=[b for b in balls if b.get('visibility')=='VISIBLE']
            b=np.full(2,np.nan,np.float32)
            if len(balls)==1 and balls[0].get('x') is not None and balls[0].get('y') is not None:
                b[:]=[.5+float(balls[0]['x'])/L,.5+float(balls[0]['y'])/W]
            if fr in part['seen']:
                stats['duplicate_rows']+=1;i=part['seen'][fr]
                if not np.array_equal(p,part['P'][i],equal_nan=True) or not np.array_equal(b,part['B'][i],equal_nan=True):
                    part['P'][i][:]=np.nan;part['B'][i][:]=np.nan;stats['duplicate_geometry_conflict']+=1
                continue
            part['seen'][fr]=len(part['t']);part['t'].append(t);part['P'].append(p);part['B'].append(b)
    result={'pitch':np.array([L,W],np.float32),'team':team,'gk':gk}
    per_reports=[]
    for per,part in sorted(parts.items()):
        t=np.array(part['t']);order=np.argsort(t);t=t[order]
        assert np.all(np.diff(t)>0),'Non-unique video time after frame dedup'
        rawP=np.asarray(part['P'])[order];rawB=np.asarray(part['B'])[order]
        grid=np.arange(int(np.ceil(t[0]*FPS-1e-7)),int(np.floor(t[-1]*FPS+1e-7))+1)/FPS
        P,coverage=resample(t,rawP,grid);B,_=resample(t,rawB,grid)
        res=np.array(list(residuals.get(per,{}).values()))
        med=float(np.median(res)) if len(res) else None
        p95=float(np.quantile(np.abs(res-med),.95)) if len(res) else None
        aligned=len(res)>=20 and abs(med)<=120 and p95<=.5
        # Shared IDs expose systematic one-source-frame offsets. Correct even a
        # 33 ms offset; it matters on the 40 ms target grid. Ignore sub-ms rounding.
        shift=med if aligned and abs(med)>.001 else 0.
        native,ctrl,labstats=labels_from_events(events,per,grid,shift)
        guard=maximum_filter1d((~coverage).astype(np.int8),size=101,mode='constant')>0
        enough=(np.isfinite(P[:,team==0,0]).sum(1)>=6)&(np.isfinite(P[:,team==1,0]).sum(1)>=6)
        mask=coverage&~guard&enough
        if not aligned: mask[:]=False
        native[~mask]=X;ctrl[~mask]=X
        tag=f'h{per}'
        result.update({tag+'_players':P,tag+'_ball':B,tag+'_native':native,tag+'_labels':ctrl,
                       tag+'_time':grid,tag+'_coverage':coverage,tag+'_training_mask':mask})
        both=(native<2)&(ctrl<2)
        per_reports.append(dict(period=per,n=len(grid),start_seconds=float(grid[0]),end_seconds=float(grid[-1]),
            shared_alignment_events=len(res),alignment_offset_median=med,alignment_residual_p95=p95,
            applied_shift=shift,alignment_accepted=aligned,coverage=float(coverage.mean()),
            usable_geometry=float(mask.mean()),ball_seen=float(np.isfinite(B[:,0]).mean()),
            native_counts=np.bincount(native,minlength=4).tolist(),controlled_counts=np.bincount(ctrl,minlength=4).tolist(),
            native_control_disagree=float(np.mean(native[both]!=ctrl[both])) if both.any() else None,**labstats))
    result['periods']=np.array(sorted(parts),np.int8)
    info=dict(match=match,version=VERSION,source_fps=meta.get('fps'),raw_counts=dict(stats),periods=per_reports,
              event_periods=sorted({int((r.get('gameEvents') or {}).get('period') or 0) for r in events}))
    # Uncompressed cache is faster for repeated feature creation; all stays local.
    tmp=dest.with_suffix('.part.npz');np.savez(tmp,**result);tmp.replace(dest)
    report.write_text(json.dumps(info,indent=2),encoding='utf-8')
    return dest,info


if __name__=='__main__':
    import sys
    path,info=parse(sys.argv[1],refresh='--refresh' in sys.argv)
    print(json.dumps(info,indent=2),flush=True)
