import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import seq_pff as seq


class SequenceTests(unittest.TestCase):
    def test_scaler_sources_equally_weighted_and_missing_safe(self):
        a=np.array([[0.,np.nan],[2.,np.nan]])
        b=np.repeat([[10.,3.]],100,axis=0)
        sc=seq.Scaler.fit([a,b])
        np.testing.assert_allclose(sc.mean,[5.5,3.])
        z=sc.transform(np.array([[np.nan,np.inf]]))
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_equal(z[0,2:],[1,1,1])

    def test_window_cannot_wrap_across_half_boundaries(self):
        x=np.ones((20,2,3),np.float32)*7
        sc=seq.Scaler(np.zeros(3),np.ones(3))
        z=seq.window(x,0,0,sc)
        self.assertTrue((z[:seq.HALO]==0).all())
        self.assertTrue((z[seq.HALO:seq.HALO+20,-1]==1).all())
        self.assertTrue((z[seq.HALO+20:]==0).all())

    def test_source_loss_not_dominated_by_more_live_frames(self):
        pred=torch.tensor([[0.,0.,0.],[2.,2.,2.]],requires_grad=True)
        target=torch.ones_like(pred);mask=torch.tensor([[1.,0.,0.],[1.,1.,1.]])
        loss=seq.balanced_loss(pred,target,mask,torch.tensor([0,1]))
        expected=.5*(torch.nn.functional.softplus(torch.tensor(0.))+
                     torch.nn.functional.softplus(torch.tensor(-2.)))
        self.assertAlmostEqual(float(loss.detach()),float(expected),places=6)
        loss.backward();self.assertEqual(float(pred.grad[0,1]),0.)

    def test_prediction_covers_tail_and_is_side_symmetric(self):
        class Pointwise(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,x):return x[:,:,0]*self.weight
        raw=np.random.default_rng(2).normal(size=(601,2,3)).astype(np.float32)
        sc=seq.Scaler(np.zeros(3),np.ones(3));net=Pointwise()
        p=seq.predict_raw(net,sc,raw,1201)
        q=seq.predict_raw(net,sc,raw[:,::-1],1201)
        expected=.5*(raw[:,0,0]-raw[:,1,0])
        expected=1/(1+np.exp(-np.interp(np.arange(1201),np.arange(601)*2,expected)))
        np.testing.assert_allclose(p,expected,atol=1e-6)
        np.testing.assert_allclose(p+q,1.,atol=1e-6)

    def test_both_architectures_backpropagate_masked_sequence(self):
        torch.set_num_threads(2)
        for kind in ('tcn','bigru'):
            net=seq.Net(7,kind,8,.1)
            out=net(torch.randn(2,seq.WINDOW,7))[:,seq.HALO:seq.HALO+seq.CORE]
            mask=torch.ones_like(out);mask[:,0]=0
            loss=seq.balanced_loss(out,torch.zeros_like(out),mask,torch.tensor([0,1]))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_tcn_halo_preserves_full_sequence_context_at_chunk_seams(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        raw=np.random.default_rng(1).normal(size=(601,2,3)).astype(np.float32)
        sc=seq.Scaler(np.zeros(3),np.ones(3));net=seq.Net(7,'tcn',8,0.).eval()
        blocks=[]
        for side in (0,1):
            z=sc.transform(raw[:,side])
            blocks.append(np.pad(z,((seq.HALO,seq.HALO),(0,0))))
        with torch.no_grad():
            y=net(torch.from_numpy(np.stack(blocks)))[:,seq.HALO:seq.HALO+len(raw)].numpy()
        logit=.5*(y[0]-y[1]);expected=1/(1+np.exp(-np.interp(np.arange(1201),np.arange(601)*2,logit)))
        np.testing.assert_allclose(seq.predict_raw(net,sc,raw,1201),expected,atol=2e-6)


if __name__=='__main__':unittest.main()
