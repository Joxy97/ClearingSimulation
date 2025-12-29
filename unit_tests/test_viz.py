from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from clearing.viz import plot_instrument_and_state


class TestViz(unittest.TestCase):
    def test_invalid_window_raises(self) -> None:
        R = torch.zeros((10, 2))
        Z = torch.zeros((10,), dtype=torch.long)
        with self.assertRaises(ValueError):
            plot_instrument_and_state(R, Z, i=0, S=3, start=5, end=5)
