"""
Gray code encoding/decoding utilities for market return discretization.

This module provides tools to convert continuous asset returns into
Gray-coded binary representations for Boltzmann Machine training.

Usage:
    from market_bm.utils import to_gray, from_gray, get_quantizer

    quantizer = get_quantizer()
    bits = to_gray(0.02, "AAPL", quantizer)
    value = from_gray(bits, "AAPL", quantizer)
"""

import re
import numpy as np
import pandas as pd
from typing import Union

# =============================================================================
# Constants
# =============================================================================

TICKERS = [
    'TLT', 'GLD', 'UNH', 'WMT', 'JNJ', 'PG', 'TSLA', 'META', 'NVDA', 'AMZN',
    'AAPL', 'MSFT', 'GOOGL', 'XOM', 'XLE', 'CVX', 'BRK-B', 'XLF', 'GS', 'MS',
    'JPM', 'BAC', 'QQQ', 'XLK', 'SPY', 'IWM', 'HYG', 'V', 'MA'
]
START_DATE = "2012-05-18"
END_DATE = "2026-02-01"
K = 8  # Number of quantization bins (3 bits per asset)


# =============================================================================
# Gray Code Primitives
# =============================================================================

def _gray_encode_int(n: int) -> int:
    """Convert integer to binary-reflected Gray code."""
    return n ^ (n >> 1)


def _gray_decode_int(g: int) -> int:
    """Convert Gray code back to integer."""
    n = g
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def _int_to_bits(x: int, nbits: int) -> np.ndarray:
    """Convert integer to bit array of length nbits (MSB first)."""
    return np.array([(x >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    """Convert bit array (MSB first) back to integer."""
    result = 0
    for b in bits:
        result = (result << 1) | int(b)
    return result


def _parse_gray_column(col: str) -> tuple[str, str | None, int] | None:
    """
    Parse a Gray-coded column name to extract asset, time suffix, and bit index.

    Supported formats:
        - "AAPL_g0"           -> ("AAPL", None, 0)
        - "AAPL_today_g0"     -> ("AAPL", "today", 0)
        - "AAPL_tomorrow_g2"  -> ("AAPL", "tomorrow", 2)
        - "BRK-B_today_g1"    -> ("BRK-B", "today", 1)

    Returns:
        Tuple of (asset, time_suffix, bit_index) or None if pattern doesn't match
    """
    # Pattern: ASSET[_TIMESUFFIX]_gN
    # Asset can contain letters, numbers, hyphens (e.g., BRK-B)
    pattern = r'^([A-Za-z0-9\-]+?)(?:_(today|tomorrow))?_g(\d+)$'
    match = re.match(pattern, col)
    if match:
        asset = match.group(1)
        time_suffix = match.group(2)  # None if not present
        bit_idx = int(match.group(3))
        return asset, time_suffix, bit_idx
    return None


# =============================================================================
# GrayQuantizer Class
# =============================================================================

class GrayQuantizer:
    """
    Quantizer that discretizes continuous values into bins and encodes them as Gray codes.

    Each asset has its own bin edges (learned from data via quantile binning).
    Provides methods to convert between decimal values and Gray-coded binary.
    """

    def __init__(self, K: int = 8):
        """
        Initialize the quantizer.

        Args:
            K: Number of discrete bins per asset. Should be a power of 2 for clean Gray encoding.
        """
        self.K = K
        self.nbits = int(np.ceil(np.log2(K)))
        self.edges: dict[str, np.ndarray] = {}
        self.centers: dict[str, np.ndarray] = {}
        self.assets: list[str] = []
        self._fitted = False

    def fit(self, returns: pd.DataFrame, assets: list[str] = None) -> "GrayQuantizer":
        """
        Fit the quantizer to return data, learning bin edges for each asset.

        Args:
            returns: DataFrame of returns with asset columns
            assets: List of assets to fit (uses all columns if None)

        Returns:
            self (for method chaining)
        """
        if assets is None:
            assets = list(returns.columns)

        self.assets = assets
        self.edges = {}
        self.centers = {}

        for asset in assets:
            x = returns[asset].dropna().values

            # Use quantile binning - duplicates='drop' handles tied values
            _, bin_edges = pd.qcut(x, q=self.K, labels=False, retbins=True, duplicates="drop")

            actual_bins = len(bin_edges) - 1
            if actual_bins < self.K:
                print(f"[warn] {asset}: only {actual_bins} bins created (requested {self.K})")

            self.edges[asset] = bin_edges

            # Compute bin centers for decoding (midpoint of each bin)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self.centers[asset] = centers

        self._fitted = True
        return self

    def _value_to_bin(self, value: float, asset: str) -> int:
        """Map a continuous value to its bin index for a given asset."""
        edges = self.edges[asset]
        bin_idx = np.searchsorted(edges, value, side="right") - 1
        return int(np.clip(bin_idx, 0, len(edges) - 2))

    def _bin_to_value(self, bin_idx: int, asset: str) -> float:
        """Map a bin index back to a representative value (bin center)."""
        centers = self.centers[asset]
        bin_idx = int(np.clip(bin_idx, 0, len(centers) - 1))
        return float(centers[bin_idx])

    def to_gray(self, value: Union[float, np.ndarray, pd.Series], asset: str) -> np.ndarray:
        """
        Convert decimal value(s) to Gray-coded binary for a specific asset.

        Args:
            value: Single value, numpy array, or pandas Series of return values
            asset: Asset ticker (must have been fitted)

        Returns:
            Gray-coded bits as uint8 array. Shape: (nbits,) for scalar, (n, nbits) for array
        """
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")
        if asset not in self.edges:
            raise ValueError(f"Asset '{asset}' not found. Available: {self.assets}")

        scalar_input = np.isscalar(value)
        values = np.atleast_1d(value)

        bits = np.zeros((len(values), self.nbits), dtype=np.uint8)
        for i, v in enumerate(values):
            bin_idx = self._value_to_bin(v, asset)
            gray_int = _gray_encode_int(bin_idx)
            bits[i, :] = _int_to_bits(gray_int, self.nbits)

        return bits[0] if scalar_input else bits

    def from_gray(self, bits: np.ndarray, asset: str) -> Union[float, np.ndarray]:
        """
        Convert Gray-coded binary back to decimal value(s) for a specific asset.

        Args:
            bits: Gray-coded bits as uint8 array. Shape: (nbits,) or (n, nbits)
            asset: Asset ticker (must have been fitted)

        Returns:
            Decoded value(s) - bin center(s) as float or array
        """
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")
        if asset not in self.edges:
            raise ValueError(f"Asset '{asset}' not found. Available: {self.assets}")

        bits = np.atleast_2d(bits)
        if bits.shape[1] != self.nbits:
            raise ValueError(f"Expected {self.nbits} bits, got {bits.shape[1]}")

        values = np.zeros(len(bits), dtype=np.float64)
        for i, row in enumerate(bits):
            gray_int = _bits_to_int(row)
            bin_idx = _gray_decode_int(gray_int)
            values[i] = self._bin_to_value(bin_idx, asset)

        return float(values[0]) if len(values) == 1 else values

    def encode_dataframe(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode entire DataFrame of returns to Gray-coded binary.

        Args:
            returns: DataFrame with asset columns (subset of fitted assets)

        Returns:
            Tuple of (binary_df, states_df):
                - binary_df: Gray-coded bits with columns like 'AAPL_g0', 'AAPL_g1', ...
                - states_df: Integer bin indices for each asset
        """
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")

        assets = [col for col in returns.columns if col in self.edges]

        # Build states DataFrame
        states = pd.DataFrame(index=returns.index, columns=assets, dtype="Int64")
        for asset in assets:
            for idx in returns.index:
                val = returns.loc[idx, asset]
                if pd.notna(val):
                    states.loc[idx, asset] = self._value_to_bin(val, asset)

        # Drop rows with any NaN
        states_clean = states.dropna().astype(int)

        # Build binary DataFrame
        bit_data = []
        bit_cols = []

        for asset in assets:
            bits = self.to_gray(states_clean[asset].values, asset)
            for b in range(self.nbits):
                bit_cols.append(f"{asset}_g{b}")
                bit_data.append(bits[:, b])

        binary_df = pd.DataFrame(
            np.column_stack(bit_data),
            index=states_clean.index,
            columns=bit_cols,
            dtype=np.uint8
        )

        return binary_df, states_clean

    def decode_dataframe(self, binary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode Gray-coded binary DataFrame back to return values.

        Handles multiple column naming conventions:
            - 'AAPL_g0', 'AAPL_g1', ... -> decoded column 'AAPL'
            - 'AAPL_today_g0', ... -> decoded column 'AAPL_today'
            - 'AAPL_tomorrow_g0', ... -> decoded column 'AAPL_tomorrow'

        Args:
            binary_df: DataFrame with Gray-coded columns

        Returns:
            DataFrame of decoded return values (bin centers)
        """
        if not self._fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")

        # Parse all columns and group by (asset, time_suffix)
        # Key: (asset, time_suffix) -> column_key for output
        # We only care about _g0 columns to identify unique (asset, time_suffix) pairs
        column_groups: dict[str, str] = {}  # output_col -> asset (for quantizer lookup)

        for col in binary_df.columns:
            parsed = _parse_gray_column(col)
            if parsed is None:
                continue
            asset, time_suffix, bit_idx = parsed

            # Only process _g0 to identify groups (avoid duplicates)
            if bit_idx != 0:
                continue

            # Skip if asset not in quantizer
            if asset not in self.edges:
                continue

            # Build output column name
            if time_suffix:
                output_col = f"{asset}_{time_suffix}"
            else:
                output_col = asset

            column_groups[output_col] = asset

        # Decode each group
        decoded = pd.DataFrame(index=binary_df.index, columns=list(column_groups.keys()), dtype=np.float64)

        for output_col, asset in column_groups.items():
            # Reconstruct the bit column names
            if output_col == asset:
                bit_cols = [f"{asset}_g{b}" for b in range(self.nbits)]
            else:
                # output_col is "ASSET_today" or "ASSET_tomorrow"
                time_suffix = output_col[len(asset) + 1:]  # skip "ASSET_"
                bit_cols = [f"{asset}_{time_suffix}_g{b}" for b in range(self.nbits)]

            bits = binary_df[bit_cols].values
            decoded[output_col] = self.from_gray(bits, asset)

        return decoded


# =============================================================================
# Module-level quantizer (lazy initialization)
# =============================================================================

_quantizer_cache: GrayQuantizer | None = None


def get_quantizer(returns: pd.DataFrame = None) -> GrayQuantizer:
    """
    Get a fitted GrayQuantizer instance.

    On first call, fetches market data and fits the quantizer.
    Subsequent calls return the cached instance.

    Args:
        returns: Optional pre-loaded returns DataFrame. If None, fetches from yfinance.

    Returns:
        Fitted GrayQuantizer instance
    """
    global _quantizer_cache

    if _quantizer_cache is not None:
        return _quantizer_cache

    if returns is None:
        import yfinance as yf
        prices = yf.download(
            TICKERS,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False
        )["Adj Close"]
        returns = prices.pct_change().dropna(how="all")

    _quantizer_cache = GrayQuantizer(K=K)
    _quantizer_cache.fit(returns, TICKERS)

    return _quantizer_cache


# =============================================================================
# Convenience Functions
# =============================================================================

def to_gray(value: Union[float, np.ndarray], asset: str, quantizer: GrayQuantizer = None) -> np.ndarray:
    """
    Convert decimal return value(s) to Gray-coded binary.

    Args:
        value: Return value(s) to encode
        asset: Asset ticker
        quantizer: Fitted GrayQuantizer (uses get_quantizer() if None)

    Returns:
        Gray-coded bits as uint8 array
    """
    if quantizer is None:
        quantizer = get_quantizer()
    return quantizer.to_gray(value, asset)


def from_gray(bits: np.ndarray, asset: str, quantizer: GrayQuantizer = None) -> Union[float, np.ndarray]:
    """
    Convert Gray-coded binary back to decimal return value(s).

    Args:
        bits: Gray-coded bits (shape: (nbits,) or (n, nbits))
        asset: Asset ticker
        quantizer: Fitted GrayQuantizer (uses get_quantizer() if None)

    Returns:
        Decoded return value(s) - bin centers
    """
    if quantizer is None:
        quantizer = get_quantizer()
    return quantizer.from_gray(bits, asset)


def decode_paired_dataframe(
    binary_df: pd.DataFrame,
    time_slice: str = None,
    quantizer: GrayQuantizer = None
) -> pd.DataFrame:
    """
    Decode a paired (today/tomorrow) Gray-coded DataFrame.

    Args:
        binary_df: DataFrame with columns like 'AAPL_today_g0', 'AAPL_tomorrow_g0', ...
        time_slice: Optional filter - "today", "tomorrow", or None for all
        quantizer: Fitted GrayQuantizer (uses get_quantizer() if None)

    Returns:
        DataFrame of decoded return values. Column names:
            - If time_slice=None: 'AAPL_today', 'AAPL_tomorrow', ...
            - If time_slice="today": 'AAPL', 'MSFT', ... (time suffix stripped)
            - If time_slice="tomorrow": 'AAPL', 'MSFT', ... (time suffix stripped)
    """
    if quantizer is None:
        quantizer = get_quantizer()

    # Filter columns if time_slice specified
    if time_slice:
        cols = [c for c in binary_df.columns if f"_{time_slice}_g" in c]
        binary_df = binary_df[cols]

    decoded = quantizer.decode_dataframe(binary_df)

    # Strip time suffix from column names if filtering by time_slice
    if time_slice:
        decoded.columns = [c.replace(f"_{time_slice}", "") for c in decoded.columns]

    return decoded
