import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from exp_external_ensemble import average, paired_summary
from exp_barath_ensemble import equal_probability_mean


class ExternalEnsembleTests(unittest.TestCase):
    def test_identical_operation_to_submission_ensemble(self):
        members = [[np.array(x), np.array([.35])] for x in ([.9,.1],[.4,.4],[.4,.7])]
        packed = [{10:(None,m[0]),20:(None,m[1])} for m in members]
        expected = equal_probability_mean(packed, [(1,10,11),(2,20,20)])
        actual = average(members, [2,1])
        for value, start in zip(actual, [10,20]):
            np.testing.assert_array_equal(value, expected[start][1])
        self.assertGreater(actual[0][0], .5)  # Hard-vote would choose the other team.

    def test_invalid_alignment_and_values_rejected(self):
        for members in ([[np.array([.5])]]*2, [[np.array([.5,.5])]]*3,
                        [[np.array([np.nan])]]*3, [[np.array([1.1])]]*3):
            with self.assertRaises(ValueError):
                average(members, [1])

    def test_bootstrap_preserves_match_pairing(self):
        # Identical per-match improvement has a point interval even if scores vary.
        result = paired_summary({'b':.9,'a':.3}, {'a':.2,'b':.8})
        np.testing.assert_allclose(result['ci95'], [.1,.1])
        self.assertEqual((result['wins'],result['matches']), (2,2))
        with self.assertRaises(AssertionError):
            paired_summary({'a':.5}, {'b':.5})


if __name__ == '__main__':
    unittest.main()
