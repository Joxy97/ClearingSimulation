from __future__ import annotations

import unittest

import torch

from clearing.quantization import ReturnQuantizer
from clearing.sampler import _visible_block_slices_from_config, sample_scenarios_from_float


class _RBMStub:
    def __init__(self) -> None:
        self._eval = False

    def eval(self):
        self._eval = True
        return self

    def sample_clamped(self, *, v_clamp, clamp_idx, n_samples, burn_in, thin, device):
        dev = torch.device(device)
        v = v_clamp.repeat(n_samples, 1).clone().to(device=dev)
        unclamped = [i for i in range(v.shape[1]) if i not in clamp_idx]
        if unclamped:
            rand = torch.randint(0, 2, (n_samples, len(unclamped)), device=dev, dtype=v.dtype)
            v[:, unclamped] = rand
        return v


class TestSampler(unittest.TestCase):
    def test_visible_block_slices(self) -> None:
        cfg = {"model": {"visible_blocks": {"s": 3, "r_prev": 4, "r": 4}}}
        sl = _visible_block_slices_from_config(cfg)
        self.assertEqual(sl["s"], slice(0, 3))
        self.assertEqual(sl["r_prev"], slice(3, 7))
        self.assertEqual(sl["r"], slice(7, 11))
        self.assertEqual(sl["_nv"], slice(0, 11))

    def test_sample_scenarios_shape_and_bounds(self) -> None:
        q = ReturnQuantizer(lo=torch.tensor([-1.0, -2.0]), hi=torch.tensor([1.0, 2.0]), K=2)
        cfg = {"model": {"visible_blocks": {"s": 3, "r_prev": 4, "r": 4}}}

        model = _RBMStub()
        r_prev = torch.tensor([0.0, 0.0])
        out = sample_scenarios_from_float(
            model=model,
            config=cfg,
            quantizer=q,
            z_t=1,
            r_prev_float=r_prev,
            num_samples=5,
            burn_in=1,
            thin=1,
            device="cpu",
        )

        self.assertEqual(out.shape, (5, 2))
        self.assertTrue(torch.all(out >= q.lo - 1e-6))
        self.assertTrue(torch.all(out <= q.hi + 1e-6))

    def test_sample_scenarios_with_onehot(self) -> None:
        q = ReturnQuantizer(lo=torch.tensor([-1.0, -1.0]), hi=torch.tensor([1.0, 1.0]), K=1)
        cfg = {"model": {"visible_blocks": {"s": 3, "r_prev": 2, "r": 2}}}
        model = _RBMStub()

        z_t = torch.tensor([0.0, 1.0, 0.0])
        r_prev = torch.tensor([0.1, -0.1])

        out = sample_scenarios_from_float(
            model=model,
            config=cfg,
            quantizer=q,
            z_t=z_t,
            r_prev_float=r_prev,
            num_samples=3,
            burn_in=1,
            thin=1,
            device="cpu",
        )

        self.assertEqual(out.shape, (3, 2))

    def test_sample_scenarios_batch_requires_onehot(self) -> None:
        q = ReturnQuantizer(lo=torch.tensor([-1.0, -1.0]), hi=torch.tensor([1.0, 1.0]), K=1)
        cfg = {"model": {"visible_blocks": {"s": 3, "r_prev": 2, "r": 2}}}
        model = _RBMStub()

        r_prev = torch.zeros((2, 2))
        with self.assertRaises(ValueError):
            sample_scenarios_from_float(
                model=model,
                config=cfg,
                quantizer=q,
                z_t=1,
                r_prev_float=r_prev,
                num_samples=2,
                burn_in=1,
                thin=1,
                device="cpu",
            )
