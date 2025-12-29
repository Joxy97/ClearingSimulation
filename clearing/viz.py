from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_instrument_and_state(
    R: torch.Tensor,
    Z: torch.Tensor,
    i: int,
    S: int,
    start: int | None = None,
    end: int | None = None,
) -> None:
    """
    Plot returns of instrument i and market state over time.
    Optionally restrict to time window [start:end).
    """
    T = R.shape[0]

    if start is None:
        start = 0
    if end is None or end > T:
        end = T

    if not (0 <= start < end <= T):
        raise ValueError(f"Invalid window: start={start}, end={end}, T={T}")

    t = torch.arange(start, end)

    r_i = R[start:end, i].cpu()
    z = Z[start:end].cpu()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(t, r_i, linewidth=1)
    ax1.axhline(0.0)
    ax1.set_ylabel(f"Return (instrument {i})")
    ax1.set_title(f"Instrument {i}: Returns and Market State")

    ax2.step(t, z, where="post")
    ax2.set_yticks(range(S))
    ax2.set_ylabel("Market state")
    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.show()
