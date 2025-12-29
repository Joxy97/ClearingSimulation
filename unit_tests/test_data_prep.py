from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd
import torch

from clearing.data_prep import generate_dataset, load_quantizer, save_dataset_csv, save_quantizer
from clearing.market import make_default_params


class TestDataPrep(unittest.TestCase):
    def test_generate_and_save_dataset(self) -> None:
        device = "cpu"
        dtype = torch.float32
        params = make_default_params(N=2, S=3, F=1, device=device, dtype=dtype)

        Z, R, D_all, D_bits, q = generate_dataset(
            T=10,
            params=params,
            K=2,
            q_low=0.0,
            q_high=1.0,
            device=device,
            dtype=dtype,
            seed=123,
        )

        self.assertEqual(Z.shape, (10,))
        self.assertEqual(R.shape, (10, 2))
        self.assertEqual(D_all.shape, (10, params.S + 2 * params.N))
        self.assertEqual(D_bits.shape, (10, params.S + 2 * params.N * 2))

        self.assertTrue(torch.all(D_bits >= 0.0))
        self.assertTrue(torch.all(D_bits <= 1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            q_path = os.path.join(tmpdir, "quantizer.pt")

            save_dataset_csv(D_bits, csv_path)
            save_quantizer(q, q_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(q_path))

            df = pd.read_csv(csv_path)
            self.assertEqual(df.shape, (10, params.S + 2 * params.N * 2))

            q_loaded = load_quantizer(q_path)
            self.assertTrue(torch.allclose(q_loaded.lo, q.lo))
            self.assertTrue(torch.allclose(q_loaded.hi, q.hi))
            self.assertEqual(q_loaded.K, q.K)
