"""Software tests only; tiny tensors are not MRI/model benchmarks."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

from brain_mri_world_model import clean_features as clean

HAS_TORCH = importlib.util.find_spec("torch") is not None


class CleanInterfaceTests(unittest.TestCase):
    def test_import_and_help_do_not_load_tensor_backend(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys;from brain_mri_world_model import clean_features;assert 'torch' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m", "brain_mri_world_model.clean_features", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_clean_interface_has_no_noise_rng_calls(self):
        tree = ast.parse(Path(clean.__file__).read_text())
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))]
        self.assertFalse(set(calls) & {"randn", "randn_like", "manual_seed", "Generator", "fixed_seed"})


@unittest.skipUnless(HAS_TORCH, "Torch is required for numerical software tests")
class CleanTensorTests(unittest.TestCase):
    def test_only_valid_tokens_contribute(self):
        import torch
        hidden = torch.tensor([[[1., 2.], [500., 500.], [5., 6.], [-500., -500.]]])
        mask = torch.tensor([[[[True, False, True, False]]]])
        result = clean.mean_valid_tokens(hidden, mask)
        torch.testing.assert_close(result, torch.tensor([[3., 4.]]), rtol=0, atol=0)
        self.assertEqual(result.dtype, torch.float32)
        with self.assertRaises(ValueError):
            clean.mean_valid_tokens(hidden, torch.zeros_like(mask))
        with self.assertRaises(ValueError):
            clean.mean_valid_tokens(hidden.double(), mask)

    def test_direct_latent_zero_timestep_and_hook_cleanup(self):
        import torch

        class TinyForwardFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(15)])
                self.received = None
            def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict):
                self.received = (hidden_states, timestep.clone(), encoder_hidden_states)
                value = hidden_states
                for block in self.blocks:
                    value = block(value)
                return (value,)

        model = TinyForwardFixture().eval().requires_grad_(False)
        latent = torch.tensor([[[1., 2.], [500., 500.], [5., 6.], [-500., -500.]]])
        mask = torch.tensor([[[[True, False, True, False]]]])
        embeddings = torch.zeros((1, 2, 3))
        result = clean.forward_clean(model, latent, mask, embeddings)
        self.assertIs(model.received[0], latent)
        self.assertIs(model.received[2], embeddings)
        torch.testing.assert_close(model.received[1], torch.zeros(1), rtol=0, atol=0)
        torch.testing.assert_close(result, torch.tensor([3., 4.]), rtol=0, atol=0)
        self.assertEqual(len(model.blocks[14]._forward_hooks), 0)
        with self.assertRaises(ValueError):
            clean.forward_clean(model, latent, torch.zeros_like(mask), embeddings)
        self.assertEqual(len(model.blocks[14]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
