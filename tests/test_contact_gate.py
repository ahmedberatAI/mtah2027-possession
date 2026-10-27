"""Semantic regression checks for synthetic contacts, without match fixtures."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from exp_anchor import deduplicate_contacts, evidence
import exp_anchor
import pandas as pd
import exp_ctrl2 as e2
import exp_data_gate as gate
from exp_contact_gate import corrupt_contacts, typed_evidence


class ContactTests(unittest.TestCase):
    def test_recovery_evidence_is_after_contact(self):
        e = typed_evidence(np.array([[20, 0, 1]]), 50)
        self.assertEqual(e[10], 0)
        self.assertEqual(e[30], 1)
        release = typed_evidence(np.array([[20, 0, -1]]), 50)
        self.assertEqual(release[10], 1)
        self.assertEqual(release[30], 0)
    def test_parallel_decode_matches_serial(self):
        rng = np.random.default_rng(27)
        pd_ = [rng.uniform(.01, .99, 700) for _ in range(4)]
        ph = [rng.uniform(.01, .99, 700) for _ in range(4)]
        serial = e2.decode_all(gate.DEC, pd_, ph)
        parallel = gate.decode_all(pd_, ph)
        for a, b in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)
    def test_dedup_preserves_opposing_contacts(self):
        a = np.array([[10, 0], [10, 1], [12, 0], [14, 0]])
        np.testing.assert_array_equal(deduplicate_contacts(a), [[10, 0], [10, 1], [14, 0]])

    def test_evidence_clips_and_cancels(self):
        a = np.array([[5, 0], [5, 0], [5, 1]])
        e = evidence(a, 10, before=1, after=1)
        np.testing.assert_array_equal(e, [0, 0, 0, 0, 1, 1, 1, 0, 0, 0])

    def test_precision_and_team_accuracy_are_separate(self):
        a = np.column_stack((np.arange(1000, 9000), np.zeros(8000, dtype=int)))
        out, counts = corrupt_contacts(a, 10000, np.random.default_rng(0), .7, .7, .95, 0)
        self.assertAlmostEqual(counts['true'] / len(out), .7, places=3)
        self.assertLess(abs(counts['wrong_team'] / counts['true'] - .05), .015)
        self.assertTrue(np.all((out[:, 0] >= 0) & (out[:, 0] < 10000)))

    def test_perfect_is_identity_in_bounds(self):
        a = np.array([[0, 0], [9, 1]])
        out, _ = corrupt_contacts(a, 10, np.random.default_rng(4), 1, 1, 1, 0)
        np.testing.assert_array_equal(a, out)

    def test_contact_extraction_adds_recovery_not_lost_duel(self):
        ev = pd.DataFrame([
            dict(period=1, type='PASS', start=10, end=20, team=0, tokens=frozenset()),
            dict(period=1, type='RECOVERY', start=30, end=30, team=1, tokens=frozenset({'INTERCEPTION'})),
            dict(period=1, type='CHALLENGE', start=40, end=40, team=0, tokens=frozenset({'LOST'})),
            dict(period=1, type='CHALLENGE', start=50, end=50, team=1, tokens=frozenset({'WON'})),
        ])
        with patch.object(exp_anchor.ms, 'events', return_value=ev), patch.object(
                exp_anchor.ms, 'parse', return_value={'h1_n0': np.array([0])}):
            b = dict(group='g1', key='metrica/g1/s0/h1')
            np.testing.assert_array_equal(exp_anchor.metrica_anchors(b), [[10, 0]])
            np.testing.assert_array_equal(exp_anchor.metrica_anchors(b, True), [[10, 0], [20, 0], [30, 1], [50, 1]])


if __name__ == '__main__':
    unittest.main()
