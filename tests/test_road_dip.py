from pathlib import Path
import unittest

import torch

from road_dip.config import load_config, validate_config
from road_dip.data import remap_pair_labels
from road_dip.models.prototype import PrototypeBank, build_prototypes, prototype_logits


REPO_ROOT = Path(__file__).resolve().parents[1]


class RoadDIPTests(unittest.TestCase):
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

    def test_prototype_classifier_and_bank_ignore_absent_classes(self):
        features = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])
        labels = torch.tensor([[[0, 0]]])
        prototypes = build_prototypes(features, labels, [0])
        logits = prototype_logits(features, prototypes)
        self.assertEqual(prototypes.shape, (1, 2))
        self.assertEqual(logits.shape, (1, 1, 1, 2))

        bank = PrototypeBank.empty([0, 1], feature_dim=2, device=torch.device("cpu"))
        bank.update(features, labels)
        self.assertEqual(bank.valid.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
