import math
import unittest

import numpy as np
import torch

from ddigat.utils.class_weights import compute_class_priors, compute_tail_class_ids


class LogitAdjustmentSmokeTest(unittest.TestCase):
    def test_log_priors_are_finite_and_logits_adjust(self) -> None:
        counts = np.array([50, 10, 1, 0, 0], dtype=np.int64)
        priors, log_priors = compute_class_priors(counts, eps=1e-12)

        self.assertTrue(np.isfinite(log_priors).all())
        self.assertAlmostEqual(float(np.sum(priors)), 1.0, places=10)

        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
        log_pi_t = torch.tensor(log_priors, dtype=logits.dtype)
        logits_adj = logits + 1.0 * log_pi_t
        self.assertFalse(torch.allclose(logits, logits_adj))

    def test_tail_size_matches_expected_k(self) -> None:
        num_classes = 86
        counts = np.arange(num_classes, dtype=np.int64)
        tail_ids = compute_tail_class_ids(counts, fraction=0.2, include_zero_count=True)
        expected_k = int(math.ceil(0.2 * num_classes))
        self.assertEqual(int(tail_ids.size), expected_k)


if __name__ == "__main__":
    unittest.main()
