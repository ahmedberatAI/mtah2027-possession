import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bench
from exp_controlled_dfl import label_stats


def convert(lab,owners):
    n=len(lab);players=np.full((n,3,2),.9,np.float32)
    ball=np.zeros((n,4),np.float32);ball[:,:2]=.5
    players[np.arange(n),owners,:]=.5
    return bench.trk_ctrl(np.asarray(lab,np.int8),players,ball,np.array([0,1,1]),(100.,64.))


class ControlledDFLTests(unittest.TestCase):
    def test_uncontrolled_short_turnover_is_held(self):
        labels=np.repeat([0,1,0],[10,10,10]);owners=labels.copy()
        out=convert(labels,owners)
        np.testing.assert_array_equal(out,np.zeros(30))
        s=label_stats(labels,out)
        self.assertEqual(s['changed'],10);self.assertEqual(s['controlled_changes'],0)

    def test_two_players_or_25_owner_frames_preserve_control(self):
        labels=np.repeat([0,1,0],[10,10,30]);owners=labels.copy();owners[15:20]=2
        np.testing.assert_array_equal(convert(labels,owners),labels)
        labels=np.repeat([0,1,0],[10,30,30]);owners=np.zeros(70,dtype=int)
        owners[10:23]=1;owners[28:40]=1  # 25 nonconsecutive owner frames, as existing proxy specifies.
        np.testing.assert_array_equal(convert(labels,owners),labels)

    def test_restart_is_not_inherited_across_dead_ball(self):
        labels=np.repeat([0,2,1,0],[10,5,10,10]);owners=np.where(labels==1,1,0)
        out=convert(labels,owners)
        np.testing.assert_array_equal(out[10:25],labels[10:25])
        self.assertEqual(label_stats(labels,out)['live'],30)

    def test_mask_or_alignment_changes_rejected(self):
        for new in (np.array([0,1,3]),np.array([0,2,2]),np.array([0,2])):
            with self.assertRaises(ValueError):label_stats(np.array([0,2,3]),new)


if __name__=='__main__':unittest.main()
