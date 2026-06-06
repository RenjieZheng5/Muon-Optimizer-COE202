import unittest

import torch
from torch import nn

from muon_project.data import build_loaders
from muon_project.models import build_model
from muon_project.optim import Muon, is_muon_parameter, split_muon_parameters
from train import make_optimizer, set_seed


class MuonProjectTests(unittest.TestCase):
    def test_parameter_classifier_exclusion(self):
        model = build_model("mlp", (1, 28, 28), 10)
        muon_params, aux_params, muon_names = split_muon_parameters(model)
        self.assertGreater(len(muon_params), 0)
        self.assertGreater(len(aux_params), 0)
        self.assertTrue(all("classifier" not in name for name in muon_names))
        self.assertFalse(is_muon_parameter("classifier.weight", model.classifier.weight))

    def test_muon_rejects_vector_parameter(self):
        param = nn.Parameter(torch.ones(4))
        param.grad = torch.ones_like(param)
        optimizer = Muon([param], lr=1e-3)
        with self.assertRaises(ValueError):
            optimizer.step()

    def test_one_batch_adamw_and_muon(self):
        for optimizer_name in ["adamw", "muon"]:
            with self.subTest(optimizer=optimizer_name):
                set_seed(0)
                loader, _, info = build_loaders(
                    "fake",
                    batch_size=16,
                    seed=0,
                    subset_train=32,
                    subset_val=16,
                )
                model = build_model("mlp", info.input_shape, info.num_classes)

                class Args:
                    optimizer = optimizer_name
                    lr = 1e-3
                    aux_lr = 1e-3
                    weight_decay = 0.0
                    momentum = 0.95
                    ns_steps = 3
                    ns_coeffs = "muon"

                optimizer, _ = make_optimizer(Args, model)
                criterion = nn.CrossEntropyLoss()
                x, y = next(iter(loader))
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
