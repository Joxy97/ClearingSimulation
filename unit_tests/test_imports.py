from __future__ import annotations

import importlib
import os
import unittest


class TestImports(unittest.TestCase):
    def test_required_dependencies_importable(self) -> None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        required = [
            "torch",
            "numpy",
            "pandas",
            "matplotlib",
            "dimod",
            "boltzmann",
            "dwave.samplers",
        ]
        missing = []
        for mod in required:
            try:
                importlib.import_module(mod)
            except Exception as exc:
                missing.append(f"{mod}: {exc}")
        if missing:
            self.fail("Missing required dependencies:\n" + "\n".join(missing))

    def test_clearing_modules_importable(self) -> None:
        modules = [
            "clearing.market",
            "clearing.quantization",
            "clearing.sampler",
            "clearing.trading",
            "clearing.margin",
            "clearing.qubo",
            "clearing.logger",
            "clearing.simulation",
            "clearing.data_prep",
            "clearing.viz",
        ]
        missing = []
        for mod in modules:
            try:
                importlib.import_module(mod)
            except Exception as exc:
                missing.append(f"{mod}: {exc}")
        if missing:
            self.fail("Failed to import clearing modules:\n" + "\n".join(missing))

    def test_optional_dependencies(self) -> None:
        optional = ["dwave.system"]
        for mod in optional:
            try:
                importlib.import_module(mod)
            except Exception:
                self.skipTest(f"Optional dependency not available: {mod}")
