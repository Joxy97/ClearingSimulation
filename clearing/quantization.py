from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReturnQuantizer:
    """
    Uniform per-instrument quantizer:
      r_i in [lo_i, hi_i] -> idx in {0..(2^K-1)} -> K bits
    """

    lo: torch.Tensor
    hi: torch.Tensor
    K: int

    @property
    def levels(self) -> int:
        return 1 << self.K


@torch.no_grad()
def fit_return_quantizer(
    R: torch.Tensor,
    K: int,
    q_low: float = 0.001,
    q_high: float = 0.999,
    eps: float = 1e-8,
) -> ReturnQuantizer:
    """
    Compute per-instrument quantization ranges using quantiles.
    """
    if R.ndim != 2:
        raise ValueError("R must be [T, N].")

    lo = torch.quantile(R, q_low, dim=0)
    hi = torch.quantile(R, q_high, dim=0)

    hi = torch.maximum(hi, lo + eps)

    return ReturnQuantizer(lo=lo, hi=hi, K=K)


@torch.no_grad()
def encode_returns(
    r: torch.Tensor,
    q: ReturnQuantizer,
) -> torch.Tensor:
    """
    Quantize r to indices then pack indices into K bits (LSB-first).
    Output bits are float32 {0,1} (RBM friendly).
    """
    squeeze = False
    if r.ndim == 1:
        r = r[None, :]
        squeeze = True
    if r.ndim != 2:
        raise ValueError("r must be [N] or [B,N].")

    B, N = r.shape
    if q.lo.numel() != N:
        raise ValueError("Quantizer N mismatch.")

    lo = q.lo.to(device=r.device, dtype=r.dtype)
    hi = q.hi.to(device=r.device, dtype=r.dtype)

    u = (r - lo[None, :]) / (hi - lo)[None, :]
    u = torch.clamp(u, 0.0, 1.0)

    L = q.levels
    idx = torch.round(u * (L - 1)).to(torch.int64)

    bitpos = torch.arange(q.K, device=r.device, dtype=torch.int64)
    bits = ((idx[..., None] >> bitpos) & 1).to(torch.float32)

    bits = bits.reshape(B, N * q.K)
    return bits[0] if squeeze else bits


@torch.no_grad()
def decode_returns(
    bits: torch.Tensor,
    q: ReturnQuantizer,
) -> torch.Tensor:
    """
    Decode K bits per instrument into bin index, then map to bin center in [lo,hi].
    """
    squeeze = False
    if bits.ndim == 1:
        bits = bits[None, :]
        squeeze = True
    if bits.ndim != 2:
        raise ValueError("bits must be [N*K] or [B,N*K].")

    B, NK = bits.shape
    N = q.lo.numel()
    if NK != N * q.K:
        raise ValueError(f"bits length must be N*K = {N*q.K}, got {NK}.")

    b = (bits > 0.5).to(torch.int64).reshape(B, N, q.K)

    bitpos = torch.arange(q.K, device=bits.device, dtype=torch.int64)
    idx = (b * (1 << bitpos)).sum(dim=-1)

    lo = q.lo.to(device=bits.device, dtype=torch.float32)
    hi = q.hi.to(device=bits.device, dtype=torch.float32)

    L = q.levels
    u = idx.to(torch.float32) / float(L - 1)
    r = lo[None, :] + u * (hi - lo)[None, :]

    return r[0] if squeeze else r


@torch.no_grad()
def encode_d(
    d: torch.Tensor,
    S: int,
    q: ReturnQuantizer,
) -> torch.Tensor:
    squeeze = False
    if d.ndim == 1:
        d = d[None, :]
        squeeze = True
    if d.ndim != 2:
        raise ValueError("d must be [S+2N] or [B,S+2N].")

    B, D = d.shape
    N = q.lo.numel()
    if D != S + 2 * N:
        raise ValueError(f"d length must be S+2N = {S+2*N}, got {D}.")

    state = d[:, :S]
    r_prev = d[:, S:S + N]
    r_t = d[:, S + N:S + 2 * N]

    bits_prev = encode_returns(r_prev, q)
    bits_t = encode_returns(r_t, q)

    d_bits = torch.cat([state.to(torch.float32), bits_prev, bits_t], dim=1)
    return d_bits[0] if squeeze else d_bits


@torch.no_grad()
def decode_d(
    d_bits: torch.Tensor,
    S: int,
    q: ReturnQuantizer,
) -> torch.Tensor:
    squeeze = False
    if d_bits.ndim == 1:
        d_bits = d_bits[None, :]
        squeeze = True
    if d_bits.ndim != 2:
        raise ValueError("d_bits must be [..] or [B,..].")

    B, Db = d_bits.shape
    N = q.lo.numel()
    expected = S + 2 * N * q.K
    if Db != expected:
        raise ValueError(f"d_bits length must be {expected}, got {Db}.")

    state = (d_bits[:, :S] > 0.5).to(torch.float32)

    bits_prev = d_bits[:, S:S + N * q.K]
    bits_t = d_bits[:, S + N * q.K:]

    r_prev = decode_returns(bits_prev, q).to(torch.float32)
    r_t = decode_returns(bits_t, q).to(torch.float32)

    d_hat = torch.cat([state, r_prev, r_t], dim=1)
    return d_hat[0] if squeeze else d_hat
