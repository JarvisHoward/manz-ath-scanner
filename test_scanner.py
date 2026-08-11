import json
import tempfile
import unittest
from pathlib import Path

from ath_scanner import ScanConfig, generate_demo_data, run, scan_latest


class ScannerSmokeTest(unittest.TestCase):
    def test_demo_generates_pattern_candidates(self):
        candidates, _ = scan_latest(generate_demo_data(), ScanConfig())
        patterns = {candidate.pattern for candidate in candidates}
        self.assertIn("Fast Ball Short", patterns)
        self.assertIn("Infield Fly Short", patterns)
        self.assertIn("Switch Hitter Short", patterns)

    def test_candidates_have_positive_risk_and_size(self):
        candidates, _ = scan_latest(generate_demo_data(), ScanConfig(risk_dollars=1000, minimum_reward_to_risk=1.0))
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertGreater(candidate.risk_per_share, 0)
            self.assertGreater(candidate.shares, 0)
            self.assertLess(candidate.target, candidate.entry)
            self.assertGreater(candidate.stop, candidate.entry)
            self.assertGreaterEqual(candidate.target_rr, 1.0)

    def test_target_rr_is_not_fixed(self):
        candidates, _ = scan_latest(generate_demo_data(), ScanConfig())
        ratios = {candidate.target_rr for candidate in candidates}
        self.assertTrue(ratios)
        self.assertNotEqual(ratios, {1.5})

    def test_hermes_handoff_is_review_gated(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run(None, output, ScanConfig(), demo=True)
            handoff = json.loads((output / "hermes_candidates.json").read_text(encoding="utf-8"))
            self.assertFalse(handoff["order_transmission_enabled"])
            self.assertTrue(handoff["requires_human_approval"])
            self.assertEqual(handoff["candidate_count"], len(handoff["candidates"]))


if __name__ == "__main__":
    unittest.main()
