from pathlib import Path
import unittest

import torch

from road_dip.config import load_config, validate_config
from road_dip.data import remap_pair_labels
from road_dip.engine import safe_cross_entropy
from road_dip.models.prototype import PrototypeBank, build_prototypes, prototype_logits


REPO_ROOT = Path(__file__).resolve().parents[1]


class RoadDIPTests(unittest.TestCase):
    def test_safe_cross_entropy_handles_all_ignore_crop(self):
        logits = torch.randn(1, 3, 4, 4, requires_grad=True)
        target = torch.full((1, 4, 4), 255, dtype=torch.long)

        loss = safe_cross_entropy(logits, target)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_all_road_configs_are_valid(self):
        configs = sorted((REPO_ROOT / "configs" / "road").glob("*/*.yaml"))
        configs = [path for path in configs if path.parent.name != "_base_"]
        self.assertEqual(len(configs), 10)
        for path in configs:
            validate_config(load_config(path))

    def test_pair_label_remapping(self):
        source = torch.tensor([[0, 2, 8, 255]])
        support = torch.tensor([[8, 2, 7, 255]])
        source_out, support_out = remap_pair_labels(source, support, [2, 8])
        self.assertEqual(source_out.tolist(), [[255, 0, 1, 255]])
        self.assertEqual(support_out.tolist(), [[1, 0, 255, 255]])

        # DIP defines the episode from support classes, including a class that
        # may be absent from the current source query.
        source_out, support_out = remap_pair_labels(source, support, [2, 7, 8])
        self.assertEqual(source_out.tolist(), [[255, 0, 2, 255]])
        self.assertEqual(support_out.tolist(), [[2, 0, 1, 255]])

    def test_prototype_classifier_and_bank_ignore_absent_classes(self):
        features = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
        labels = torch.tensor([[[0, 0]]])
        prototypes = build_prototypes(features, labels, [0])
        logits = prototype_logits(features, prototypes)
        self.assertEqual(prototypes.shape, (1, 2))
        self.assertEqual(logits.shape, (1, 1, 1, 2))

        bank = PrototypeBank.empty(
            [0, 1], feature_dim=2, device=torch.device("cpu"), max_shots=2
        )
        bank.update(features, labels)
        bank.update(features * 3, labels)
        bank.update(features * 10, labels)
        self.assertEqual(bank.valid.tolist(), [True, False])
        self.assertEqual(bank.counts.tolist(), [2.0, 0.0])
        self.assertTrue(torch.allclose(bank.prototypes[0], torch.tensor([2.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
