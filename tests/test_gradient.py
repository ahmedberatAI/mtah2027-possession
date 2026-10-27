import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import bz2
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import gradient as g


def row(i,t,team=0,event='IT',end=None,game='OTB',**extra):
    pe=dict(possessionEventType=event,**extra)
    return dict(gameEventId=i,possessionEventId=i,startTime=t,endTime=t if end is None else end,
                eventTime=t,gameEvents=dict(period=1,gameEventType=game,homeTeam=team==0,setpieceType='O'),
                possessionEvents=pe)


class GradientTests(unittest.TestCase):
    def test_stream_parser_deduplicates_and_never_uses_smoothed_ball(self):
        tmp_root=g.ROOT/'cache/test_gradient';tmp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            base=Path(tmp)/'data';cache=Path(tmp)/'cache'
            for k in ('metadata','rosters','events','tracking'): (base/k).mkdir(parents=True)
            meta=dict(homeTeam={'id':1},fps=30,stadium={'pitches':[{'length':105,'width':68}]})
            roster=[dict(team={'id':tm+1},shirtNumber=j,positionGroupType='GK' if j==1 else 'CM') for tm in (0,1) for j in range(1,7)]
            events=[row(i,i/30,end=(i+1)/30) for i in range(120)]
            for kind,obj in [('metadata',[meta]),('rosters',roster),('events',events)]:
                (base/kind/'1.json').write_text(json.dumps(obj))
            with bz2.open(base/'tracking/1.jsonl.bz2','wt') as f:
                for i in range(120):
                    r=dict(period=1,frameNum=i,videoTimeMs=i*1000/30,possession_event_id=i,
                           possession_event={'start_frame':i},
                           homePlayers=[dict(jerseyNum=str(j),x=-10.5,y=j) for j in range(1,7)],
                           awayPlayers=[dict(jerseyNum=str(j),x=10.5,y=j) for j in range(1,7)],
                           balls=[dict(x=0,y=0,visibility='VISIBLE')],
                           ballsSmoothed=dict(x=999,y=999),homePlayersSmoothed=[dict(x=999,y=999)])
                    f.write(json.dumps(r)+'\n')
                    if i==60: f.write(json.dumps(r)+'\n')
            with patch.object(g,'DATA',base),patch.object(g,'CACHE',cache):
                path,info=g.parse('1')
            self.assertEqual(info['raw_counts']['duplicate_rows'],1)
            self.assertTrue(info['periods'][0]['alignment_accepted'])
            with np.load(path) as d:
                self.assertEqual(len(d['h1_time']),100)
                np.testing.assert_allclose(np.diff(d['h1_time']),.04)
                np.testing.assert_allclose(d['h1_ball'],.5)
                self.assertAlmostEqual(float(d['h1_players'][0,0,0]),.4,places=6)
                self.assertTrue((d['h1_labels']==g.H).all())

    def test_resample_30_to_25_uses_time_not_row_number(self):
        t=np.arange(61)/30
        x=np.column_stack([t,2*t])
        grid=np.arange(51)/25
        y,mask=g.resample(t,x,grid)
        np.testing.assert_allclose(y[:,0],grid,atol=1e-6)
        self.assertTrue(mask.all())

    def test_no_interpolation_across_video_cut(self):
        t=np.array([0,.04,2.,2.04]);x=t[:,None]
        y,valid=g.resample(t,x,np.array([0,.02,1.,2.02]))
        self.assertFalse(valid[2]);self.assertTrue(np.isnan(y[2]).all())
        self.assertTrue(valid[[0,1,3]].all())

    def test_rebound_and_failed_trap_do_not_switch(self):
        rows=[row(1,0),row(2,1,1,'RE'),row(3,2,1,'TC',touchType='F'),row(4,3,1,'IT')]
        n,c,_=g.labels_from_events(rows,1,np.arange(0,4,.04))
        self.assertEqual(n[30],g.A);self.assertEqual(c[30],g.H)
        self.assertEqual(c[60],g.H);self.assertEqual(c[80],g.A)

    def test_aerial_duel_waits_for_control(self):
        rows=[row(1,0),row(2,1,1,'CH',challengeType='A',challengeOutcomeType='R'),row(3,2,1,'PA')]
        _,c,_=g.labels_from_events(rows,1,np.arange(0,3,.04))
        self.assertEqual(c[30],g.H);self.assertEqual(c[60],g.A)

    def test_pass_after_duel_not_backdated(self):
        a=row(1,1,1,'CH',end=2,challengeType='A')
        b=row(1,1,1,'PA',end=2);b['eventTime']=2
        self.assertEqual(g.control_time([a,b]),2)

    def test_stoppage_ends_at_kick_not_ball_placement(self):
        a=row(3,2,1,'IT',end=4);a['gameEvents']['setpieceType']='F'
        b=row(3,2,1,'PA',end=4);b['eventTime']=4;b['gameEvents']['setpieceType']='F'
        rows=[row(1,0),row(2,1,game='OUT',event=None),a,b]
        _,c,_=g.labels_from_events(rows,1,np.arange(0,5,.04))
        self.assertEqual(c[75],g.D);self.assertEqual(c[100],g.A)

    def test_invalid_group_resets_owner(self):
        a=row(2,1,1,end=2);a['gameEvents']['videoMissing']=True
        rows=[row(1,0),a,row(3,3,1)]
        _,c,_=g.labels_from_events(rows,1,np.arange(0,4,.04))
        self.assertEqual(c[40],g.X);self.assertEqual(c[65],g.X);self.assertEqual(c[80],g.A)

    def test_offset_applies_to_all_event_times(self):
        rows=[row(1,0),row(2,1,1)]
        _,a,_=g.labels_from_events(rows,1,np.arange(0,2,.04))
        _,b,_=g.labels_from_events(rows,1,np.arange(0,2,.04)+50,offset=50)
        np.testing.assert_array_equal(a,b)

    def test_conflicting_simultaneous_controls_are_unknown(self):
        _,c,_=g.labels_from_events([row(1,0),row(2,0,1)],1,np.arange(0,1,.04))
        self.assertTrue((c==g.X).all())


if __name__=='__main__': unittest.main()
