from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from boltzmann import RBM

from .quantization import ReturnQuantizer, decode_returns, encode_returns


def _visible_block_slices_from_config(config: Dict[str, Any]) -> Dict[str, slice]:
    vb = config["model"]["visible_blocks"]
    offset = 0
    slices: Dict[str, slice] = {}
    for name, size in vb.items():
        size = int(size)
        slices[name] = slice(offset, offset + size)
        offset += size
    slices["_nv"] = slice(0, offset)
    return slices


@torch.no_grad()
def sample_scenarios_from_float(
    *,
    model: RBM,
    config: Dict[str, Any],
    quantizer: ReturnQuantizer,
    z_t: Union[int, torch.Tensor],
    r_prev_float: torch.Tensor,
    num_samples: int = 1000,
    burn_in: int = 500,
    thin: int = 10,
    device: str = "cuda",
    return_bits: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Conditional scenario sampler.
    """
    model.eval()

    sl = _visible_block_slices_from_config(config)
    s_sl = sl["s"]
    rp_sl = sl["r_prev"]
    r_sl = sl["r"]
    nv = sl["_nv"].stop

    S = s_sl.stop - s_sl.start
    rprev_len = rp_sl.stop - rp_sl.start
    r_len = r_sl.stop - r_sl.start

    dev = torch.device(device)

    squeeze_r = False
    if r_prev_float.ndim == 1:
        r_prev_float = r_prev_float[None, :]
        squeeze_r = True
    if r_prev_float.ndim != 2:
        raise ValueError("r_prev_float must be [N] or [B,N].")
    B, N = r_prev_float.shape
    r_prev_float = r_prev_float.to(device=dev, dtype=torch.float32)

    K_bits = int(quantizer.K)
    if rprev_len != N * K_bits or r_len != N * K_bits:
        raise ValueError(
            f"Visible block sizes imply N*K_bits={N*K_bits}, but config has "
            f"r_prev={rprev_len}, r={r_len}. Check N or quantizer.K."
        )

    if isinstance(z_t, int):
        z_int = z_t
    elif torch.is_tensor(z_t) and z_t.ndim == 0:
        z_int = int(z_t.item())
    elif torch.is_tensor(z_t) and z_t.ndim == 1 and z_t.numel() == 1:
        z_int = int(z_t.item())
    else:
        z_int = None

    if z_int is not None:
        if B != 1:
            raise ValueError("If r_prev_float is batched (B>1), pass z_t as one-hot [B,S].")
        state_bits = F.one_hot(
            torch.tensor(z_int, device=dev),
            num_classes=S,
        ).to(torch.float32)[None, :]
    else:
        if z_t.ndim == 1:
            z_t = z_t[None, :]
        if z_t.ndim != 2 or z_t.shape[1] != S:
            raise ValueError(f"z_t one-hot must be [S] or [B,S] with S={S}.")
        if z_t.shape[0] != B:
            raise ValueError(f"Batch mismatch: z_t has B={z_t.shape[0]} but r_prev_float has B={B}.")
        state_bits = (z_t > 0.5).to(device=dev, dtype=torch.float32)

    r_prev_bits = encode_returns(r_prev_float, quantizer)
    if r_prev_bits.ndim == 1:
        r_prev_bits = r_prev_bits[None, :]
    r_prev_bits = r_prev_bits.to(device=dev, dtype=torch.float32)

    v_clamp = torch.full((B, nv), 0.5, device=dev, dtype=torch.float32)
    v_clamp[:, s_sl] = state_bits
    v_clamp[:, rp_sl] = r_prev_bits

    clamp_idx = list(range(s_sl.start, rp_sl.stop))

    if B != 1:
        raise ValueError(
            "This sampler currently supports a single conditioning context (B=1). "
            "If you want multiple contexts in parallel, loop over contexts or extend RBM.sample_clamped."
        )

    samples_v = model.sample_clamped(
        v_clamp=v_clamp,
        clamp_idx=clamp_idx,
        n_samples=num_samples,
        burn_in=burn_in,
        thin=thin,
        device=dev,
    )

    r_bits = samples_v[:, r_sl]
    R_scenarios = decode_returns(r_bits, quantizer)

    if return_bits:
        return R_scenarios, r_bits
    return R_scenarios
