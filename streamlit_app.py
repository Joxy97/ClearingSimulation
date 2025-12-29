from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt


def _blend_color(base_rgb: tuple[int, int, int], intensity: float) -> str:
    intensity = max(0.0, min(1.0, float(intensity)))
    r = int(255 * (1.0 - intensity) + base_rgb[0] * intensity)
    g = int(255 * (1.0 - intensity) + base_rgb[1] * intensity)
    b = int(255 * (1.0 - intensity) + base_rgb[2] * intensity)
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_from_value(val: float, max_abs: float, pos_rgb: tuple[int, int, int], neg_rgb: tuple[int, int, int]) -> Optional[str]:
    if np.isnan(val) or np.isnan(max_abs):
        return None
    if max_abs <= 0:
        return None
    if abs(val) <= 1e-12:
        return None
    intensity = min(abs(val) / max_abs, 1.0)
    base = pos_rgb if val > 0 else neg_rgb
    return _blend_color(base, intensity)

PHASE_ORDER = ["start", "market_move", "settlement", "market", "margin", "default", "trades", "decision", "end"]
PHASE_LABELS = {
    "start": "Start of the Day",
    "market_move": "Market Move",
    "settlement": "Settlement",
    "market": "Market Move",
    "margin": "Margins",
    "default": "Default",
    "trades": "Suggested Trades",
    "decision": "QUBO Clearing",
    "end": "End of the Day",
}

st.set_page_config(page_title="Clearing Dashboard", layout="wide")


def _torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return None


def _build_records_by_day(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    by_day: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for rec in records:
        day = int(rec.get("day", -1))
        phase = str(rec.get("phase", ""))
        by_day.setdefault(day, {})[phase] = rec
    return by_day


def _market_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("phase") not in {"market_move", "market"}:
            continue
        day = int(rec.get("day", -1))
        z_t = rec.get("z_t")
        r_t = _to_numpy(rec.get("r_t"))
        if r_t is None:
            continue
        row = {"day": day, "z_t": z_t}
        for i in range(r_t.shape[0]):
            row[f"inst_{i + 1}"] = float(r_t[i])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("day")


def _pnl_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("phase") not in {"market_move", "market"}:
            continue
        day = int(rec.get("day", -1))
        pnl = _to_numpy(rec.get("pnl"))
        if pnl is None:
            continue
        row = {"day": day}
        for i in range(pnl.shape[0]):
            row[f"cl_{i + 1}"] = float(pnl[i])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("day")


def _wealth_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    phase_priority = [
        "settlement",
        "default",
        "trades",
        "decision",
        "end",
        "market_move",
        "market",
        "start",
    ]
    phase_rank = {p: i for i, p in enumerate(phase_priority)}
    by_day: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        day = int(rec.get("day", -1))
        phase = rec.get("phase")
        if phase not in phase_rank:
            continue
        if rec.get("W") is None:
            continue
        prev = by_day.get(day)
        if prev is None or phase_rank[phase] < phase_rank[prev.get("phase")]:
            by_day[day] = rec

    rows = []
    for day, rec in by_day.items():
        W = _to_numpy(rec.get("W"))
        if W is None:
            continue
        row = {"day": day}
        for i in range(W.shape[0]):
            row[f"cl_{i + 1}"] = float(W[i])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("day")


def _format_val(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


def _matrix_table_html(
    *,
    title: str,
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    cell_color,
    cell_note,
    row_color,
    max_rows: int,
    max_cols: int,
    decimals: int = 2,
) -> str:
    rows = min(data.shape[0], max_rows)
    cols = min(data.shape[1], max_cols)

    html = [
        f"<div class='tensor-title'>{title}</div>",
        "<div class='tensor-wrap'>",
        "<table class='tensor-table'>",
        "<tr><th></th>",
    ]
    for j in range(cols):
        html.append(f"<th>{col_labels[j]}</th>")
    html.append("</tr>")

    for i in range(rows):
        r_style = ""
        if row_color is not None:
            color = row_color(i)
            if color:
                r_style = f" style='background-color:{color};'"
        html.append(f"<tr><th{r_style}>{row_labels[i]}</th>")
        for j in range(cols):
            c_style = ""
            if cell_color is not None:
                color = cell_color(i, j)
                if color:
                    c_style = f" style='background-color:{color}; color:#111827;'"
            val = _format_val(float(data[i, j]), decimals)
            note = ""
            if cell_note is not None:
                note_text = cell_note(i, j)
                if note_text:
                    note = f"<div class='cell-note'>{note_text}</div>"
            html.append(
                f"<td{c_style}><div class='cell-val'>{val}</div>{note}</td>"
            )
        html.append("</tr>")

    html.append("</table>")
    html.append("</div>")
    return "\n".join(html)


def _vector_table_html(
    *,
    title: str,
    data: np.ndarray,
    row_labels: List[str],
    max_rows: int,
    col_label: str = "value",
    decimals: int = 2,
) -> str:
    data = data.reshape(-1, 1)
    return _matrix_table_html(
        title=title,
        data=data,
        row_labels=row_labels,
        col_labels=[col_label],
        cell_color=None,
        cell_note=None,
        row_color=None,
        max_rows=max_rows,
        max_cols=1,
        decimals=decimals,
    )


def _stats_table_html(
    *,
    title: str,
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    alive: Optional[np.ndarray],
    max_rows: int,
    decimals: int = 2,
    color_rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    def row_color(i: int) -> Optional[str]:
        if alive is None:
            return None
        return "#e8e8e8" if not bool(alive[i]) else None

    def cell_color(i: int, j: int) -> Optional[str]:
        if alive is None:
            return None
        if not bool(alive[i]):
            return "#e8e8e8"
        if color_rules is None:
            return None
        col = col_labels[j]
        rule = color_rules.get(col) if color_rules else None
        if not rule:
            return None
        values = rule.get("values")
        value = float(values[i]) if values is not None else float(data[i, j])
        return _color_from_value(
            value,
            float(rule["max_abs"]),
            rule["pos_rgb"],
            rule["neg_rgb"],
        )

    return _matrix_table_html(
        title=title,
        data=data,
        row_labels=row_labels,
        col_labels=col_labels,
        cell_color=cell_color,
        cell_note=None,
        row_color=row_color,
        max_rows=max_rows,
        max_cols=len(col_labels),
        decimals=decimals,
    )


def _portfolio_table_html(
    *,
    P: np.ndarray,
    DeltaP: Optional[np.ndarray],
    x: Optional[np.ndarray],
    alive: Optional[np.ndarray],
    max_rows: int,
    max_cols: int,
) -> str:
    def row_color(i: int) -> Optional[str]:
        if alive is None:
            return None
        return "#e0e0e0" if not bool(alive[i]) else None

    def cell_color(i: int, j: int) -> Optional[str]:
        if alive is not None and not bool(alive[i]):
            return "#e0e0e0"
        if DeltaP is None:
            return None
        dp = float(DeltaP[i, j])
        if abs(dp) <= 1e-12:
            return None
        if x is None:
            return "#fff2a8"
        return "#b6f3b6" if int(x[i]) == 1 else "#f6c1c1"

    def cell_note(i: int, j: int) -> str:
        if DeltaP is None:
            return ""
        dp = float(DeltaP[i, j])
        if abs(dp) <= 1e-12:
            return ""
        sign = "+" if dp >= 0 else ""
        color = "#16a34a" if dp >= 0 else "#dc2626"
        return f"<span style='color:{color};'>{sign}{dp:.2f}</span>"

    row_labels = [f"cl_{i + 1}" for i in range(P.shape[0])]
    col_labels = [f"inst_{j + 1}" for j in range(P.shape[1])]

    return _matrix_table_html(
        title="Portfolio",
        data=P,
        row_labels=row_labels,
        col_labels=col_labels,
        cell_color=cell_color,
        cell_note=cell_note,
        row_color=row_color,
        max_rows=max_rows,
        max_cols=max_cols,
        decimals=2,
    )


def _compute_var_es(R_scenarios: np.ndarray, alpha: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    if R_scenarios.ndim != 2:
        raise ValueError("R_scenarios must be 2D [omega, N].")
    losses = -R_scenarios
    var = np.quantile(losses, alpha, axis=0)
    es = np.zeros_like(var)
    for i in range(losses.shape[1]):
        tail = losses[:, i][losses[:, i] >= var[i]]
        es[i] = tail.mean() if tail.size else var[i]
    return var, es


def _portfolio_var_es(P: np.ndarray, R_scenarios: np.ndarray, alpha: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    if P.ndim != 2:
        raise ValueError("P must be [M, N].")
    if R_scenarios.ndim != 2 or R_scenarios.shape[1] != P.shape[1]:
        raise ValueError("R_scenarios must be [omega, N] matching P.")
    losses = -(R_scenarios @ P.T)
    var = np.quantile(losses, alpha, axis=0)
    es = np.zeros_like(var)
    for i in range(losses.shape[1]):
        tail = losses[:, i][losses[:, i] >= var[i]]
        es[i] = tail.mean() if tail.size else var[i]
    return var, es


def main() -> None:
    st.markdown(
        """
        <style>
        .tensor-title { font-weight: 700; margin-bottom: 6px; }
        .tensor-wrap { overflow-x: auto; }
        table.tensor-table { border-collapse: collapse; width: 100%; }
        table.tensor-table th, table.tensor-table td {
            border: 1px solid #cfcfcf; padding: 6px 8px; text-align: center; font-size: 12px;
            width: 56px; height: 36px;
        }
        table.tensor-table { table-layout: fixed; }
        table.tensor-table th { background: transparent; font-weight: 600; }
        .cell-val { font-weight: 600; font-size: 12px; }
        .cell-note { font-size: 10px; }
        .section-label { font-weight: 700; margin-bottom: 6px; }
        div[data-testid="stVerticalBlock"]:has(#sticky-header-marker) {
            position: sticky;
            top: 0;
            z-index: 999;
            background: var(--background-color);
            padding-top: 6px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--secondary-background-color);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Clearing Dashboard")

    default_path = os.path.join("simulations", "logs", "run_log.pt")
    path = st.sidebar.text_input("run_log.pt path", value=default_path)

    if not path or not os.path.exists(path):
        st.warning("run_log.pt not found. Provide a valid path in the sidebar.")
        st.stop()

    data = _torch_load(path)
    meta = None
    if isinstance(data, dict) and "records" in data:
        records = data.get("records")
        meta = data.get("meta")
    else:
        records = data

    if not isinstance(records, list):
        st.error("run_log.pt should contain a list of records.")
        st.stop()

    by_day = _build_records_by_day(records)
    if not by_day:
        st.warning("No records found in run_log.pt.")
        st.stop()

    days = sorted(by_day.keys())

    max_clients = int(st.sidebar.number_input("Max clients", min_value=1, value=12, step=1))
    max_assets = int(st.sidebar.number_input("Max instruments", min_value=1, value=12, step=1))
    max_scenarios = int(st.sidebar.number_input("Max scenarios", min_value=1, value=8, step=1))
    with st.sidebar.expander("Simulation metadata", expanded=False):
        if isinstance(meta, dict) and meta:
            meta_order = [
                ("M", "Number of clients"),
                ("N", "Number of instruments"),
                ("T", "Number of simulation days"),
                ("Omega", "Scenarios per day"),
                ("model_run", "RBM run folder"),
                ("quantizer", "Quantizer path"),
                ("quantizer_fit_T", "Days for fitting quantizer"),
                ("device", "Device"),
                ("dtype", "Tensor dtype"),
                ("seed_market", "Market RNG seed"),
                ("seed_trade", "Trade RNG seed"),
                ("burn_in", "RBM burn-in"),
                ("thin", "RBM thinning"),
                ("qubo_solver", "QUBO solver"),
                ("budget_B", "Risk budget B"),
                ("lambda_budget", "Budget penalty lambda"),
                ("eta_risk", "Risk penalty eta"),
                ("utility_weight", "Utility weight"),
                ("trade_base_scale", "Base trade size"),
                ("trade_noise_scale", "Trade noise scale"),
                ("trade_max_abs", "Max absolute trade size"),
                ("init_collateral_buffer", "Init collateral buffer"),
                ("init_liquidity_buffer", "Init liquidity buffer"),
                ("trade_momentum_weight", "Momentum weight"),
                ("trade_gamma_exposure", "Exposure gamma"),
                ("p_trade_when_zero", "Trade prob when zero"),
                ("zero_trade_std", "Zero trade std"),
                ("alpha", "Alpha (ES)"),
                ("post_full_margin_daily", "Post full margin daily"),
            ]
            rows = []
            seen = set()
            for key, label in meta_order:
                if key in meta:
                    rows.append({"Parameter": label, "Value": meta.get(key)})
                    seen.add(key)
            for key in sorted(k for k in meta.keys() if k not in seen):
                rows.append({"Parameter": key, "Value": meta.get(key)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No metadata found in log. Re-run the simulation to include it.")

    header_container = st.container()
    with header_container:
        st.markdown("<div id='sticky-header-marker'></div>", unsafe_allow_html=True)
        day_row = st.columns([0.6, 3.4])
        with day_row[0]:
            st.markdown("**Day:**")
        with day_row[1]:
            day = st.slider(
                "Day",
                min_value=min(days),
                max_value=max(days),
                value=min(days),
                step=1,
                label_visibility="collapsed",
            )

        phases_for_day = [p for p in PHASE_ORDER if p in by_day.get(day, {})]
        if not phases_for_day:
            st.warning("No records for selected day.")
            st.stop()
        default_phase = "margin" if "margin" in phases_for_day else phases_for_day[0]
        phase_label_map = [PHASE_LABELS.get(p, p) for p in phases_for_day]
        label_to_phase = dict(zip(phase_label_map, phases_for_day))
        phase_row = st.columns([0.6, 3.4])
        with phase_row[0]:
            st.markdown("**Phase:**")
        with phase_row[1]:
            phase_label = st.radio(
                "Phase",
                options=phase_label_map,
                index=phase_label_map.index(PHASE_LABELS.get(default_phase, default_phase)),
                horizontal=True,
                label_visibility="collapsed",
            )

        phase = label_to_phase[phase_label]

    rec = by_day[day][phase]

    trades_rec = by_day.get(day, {}).get("trades", {})
    decision_rec = by_day.get(day, {}).get("decision", {})

    P = _to_numpy(rec.get("P"))
    W = _to_numpy(rec.get("W"))
    C = _to_numpy(rec.get("C"))
    alive = _to_numpy(rec.get("alive"))
    pnl = _to_numpy(rec.get("pnl"))
    M_req_cur = _to_numpy(rec.get("M_req_cur"))
    M_req_tent = _to_numpy(rec.get("M_req_tent"))
    z_t = rec.get("z_t")
    r_t = _to_numpy(rec.get("r_t"))

    if phase == "trades":
        DeltaP = _to_numpy(trades_rec.get("DeltaP")) if trades_rec else _to_numpy(rec.get("DeltaP"))
        x = None
    elif phase == "decision":
        DeltaP = _to_numpy(trades_rec.get("DeltaP")) if trades_rec else _to_numpy(rec.get("DeltaP"))
        x = _to_numpy(decision_rec.get("x")) if decision_rec else _to_numpy(rec.get("x"))
    else:
        DeltaP = None
        x = None

    deltaM = _to_numpy(rec.get("deltaM"))

    Rs = _to_numpy(rec.get("R_scenarios"))
    Rs_port = Rs
    P_port = P
    rs_trunc_note = None
    if Rs is not None and P is not None and Rs.shape[1] != P.shape[1]:
        n_assets = min(Rs.shape[1], P.shape[1])
        Rs_port = Rs[:, :n_assets]
        P_port = P[:, :n_assets]
        rs_trunc_note = f"Scenarios stored for {Rs.shape[1]} of {P.shape[1]} instruments."
    var_cur = None
    es_cur = None
    var_tent = None
    es_tent = None
    var_cur = _to_numpy(rec.get("var_cur"))
    var_tent = _to_numpy(rec.get("var_tent"))
    es_cur = M_req_cur if M_req_cur is not None else None
    es_tent = M_req_tent if M_req_tent is not None else None
    if (var_cur is None or (phase in {"trades", "decision"} and var_tent is None)) and Rs_port is not None and P_port is not None:
        try:
            var_cur_fallback, es_cur_fallback = _portfolio_var_es(P_port, Rs_port, alpha=0.99)
            if var_cur is None:
                var_cur = var_cur_fallback
            if es_cur is None:
                es_cur = es_cur_fallback
            if phase in {"trades", "decision"} and DeltaP is not None:
                var_tent_fallback, es_tent_fallback = _portfolio_var_es(P_port + DeltaP[:, : P_port.shape[1]], Rs_port, alpha=0.99)
                if var_tent is None:
                    var_tent = var_tent_fallback
                if es_tent is None:
                    es_tent = es_tent_fallback
        except Exception:
            var_cur, es_cur, var_tent, es_tent = var_cur, es_cur, var_tent, es_tent

    if alive is None and W is not None:
        alive = np.ones(W.shape[0], dtype=bool)

    default_loss_total = rec.get("default_loss_total")
    if default_loss_total is None:
        for phase_key in ("end", "settlement", "decision", "trades", "default", "margin"):
            day_rec = by_day.get(day, {}).get(phase_key)
            if day_rec is not None and day_rec.get("default_loss_total") is not None:
                default_loss_total = day_rec.get("default_loss_total")
                break
    default_loss = _to_numpy(rec.get("default_loss"))
    if default_loss is None:
        settlement_rec = by_day.get(day, {}).get("settlement")
        if settlement_rec is not None:
            default_loss = _to_numpy(settlement_rec.get("default_loss"))

    market_state_names = {0: "Calm", 1: "Volatile", 2: "Crisis"}
    with header_container:
        market_row = st.columns([1.6, 3.4])
        with market_row[0]:
            if z_t is not None:
                state_name = market_state_names.get(int(z_t), "State")
                state_color = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}.get(int(z_t), "#111827")
                st.markdown(
                    (
                        f"<div style='font-size:20px;font-weight:600;color:#ffffff;'>"
                        f"Market state: <span style='color:{state_color};'>{state_name} ({z_t})</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            if default_loss_total is not None:
                st.markdown(
                    f"<div style='font-size:14px;color:#ffffff;'>Default loss (total): {_format_val(float(default_loss_total), 2)}</div>",
                    unsafe_allow_html=True,
                )
        with market_row[1]:
            market_df = _market_dataframe(records)
            if not market_df.empty:
                market_df = market_df[market_df["day"] <= day]
                if not market_df.empty:
                    st.line_chart(market_df.set_index("day")[["z_t"]], height=120)

    row_labels_clients = []
    client_count = None
    for vec in (W, C, M_req_cur, M_req_tent, pnl, alive):
        if vec is not None:
            client_count = vec.shape[0]
            break
    if client_count is not None:
        row_labels_clients = [f"cl_{i + 1}" for i in range(client_count)]

    row_labels_instruments = []
    inst_count = None
    for vec in (r_t,):
        if vec is not None:
            inst_count = vec.shape[0]
            break
    if inst_count is not None:
        row_labels_instruments = [f"inst_{i + 1}" for i in range(inst_count)]

    top_cols = st.columns([3.2, 4, 2.4])
    with top_cols[0]:
        stats_cols = ["DL", "W", "F", "C"]
        stats_data = []
        if default_loss is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(default_loss)
        if W is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(W)
        if W is None or C is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(W - C)
        if C is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(C)

        if phase in {"margin", "default", "trades", "decision"}:
            stats_cols.append("M")
            if M_req_cur is not None:
                stats_data.append(M_req_cur)
            elif M_req_tent is not None:
                stats_data.append(M_req_tent)
            else:
                stats_data.append(np.full((client_count or 0,), np.nan))

        if phase in {"market_move", "market"}:
            stats_cols.append("PnL")
            if pnl is None:
                stats_data.append(np.full((client_count or 0,), np.nan))
            else:
                stats_data.append(pnl)

        if phase in {"trades", "decision"}:
            stats_cols.append("DeltaM")
            if deltaM is None:
                stats_data.append(np.full((client_count or 0,), np.nan))
            else:
                stats_data.append(deltaM)

        if stats_data:
            stats_matrix = np.vstack(stats_data).T
            color_rules = {}
            if "PnL" in stats_cols:
                idx = stats_cols.index("PnL")
                pnl_col = stats_matrix[:, idx]
                max_abs = float(np.nanmax(np.abs(pnl_col))) if pnl_col.size and not np.all(np.isnan(pnl_col)) else 0.0
                color_rules["PnL"] = {"pos_rgb": (34, 197, 94), "neg_rgb": (239, 68, 68), "max_abs": max_abs}
            if phase == "margin" and "M" in stats_cols and "C" in stats_cols:
                idx_m = stats_cols.index("M")
                idx_c = stats_cols.index("C")
                m_col = stats_matrix[:, idx_m]
                c_col = stats_matrix[:, idx_c]
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_diff = np.where(
                        np.isfinite(m_col) & (np.abs(m_col) > 1e-12),
                        (c_col - m_col) / m_col,
                        0.0,
                    )
                max_abs = float(np.nanmax(np.abs(pct_diff))) if pct_diff.size and not np.all(np.isnan(pct_diff)) else 0.0
                color_rules["M"] = {
                    "pos_rgb": (34, 197, 94),
                    "neg_rgb": (239, 68, 68),
                    "max_abs": max_abs,
                    "values": pct_diff,
                }
            if "DeltaM" in stats_cols:
                idx = stats_cols.index("DeltaM")
                dm_col = stats_matrix[:, idx]
                max_abs = float(np.nanmax(np.abs(dm_col))) if dm_col.size and not np.all(np.isnan(dm_col)) else 0.0
                color_rules["DeltaM"] = {"pos_rgb": (239, 68, 68), "neg_rgb": (34, 197, 94), "max_abs": max_abs}
            st.markdown(
                _stats_table_html(
                    title="Client Stats",
                    data=stats_matrix,
                    row_labels=row_labels_clients,
                    col_labels=stats_cols,
                    alive=alive,
                    max_rows=max_clients,
                    color_rules=color_rules,
                ),
                unsafe_allow_html=True,
            )

    with top_cols[1]:
        if P is not None:
            st.markdown(
                _portfolio_table_html(
                    P=P,
                    DeltaP=DeltaP,
                    x=x,
                    alive=alive,
                    max_rows=max_clients,
                    max_cols=max_assets,
                ),
                    unsafe_allow_html=True,
                )
            if phase in {"trades", "decision"}:
                st.markdown(
                    """
                    <div style='display:flex;gap:10px;font-size:12px;margin-top:6px;'>
                        <div><span style='background:#fff2a8;padding:2px 6px;border:1px solid #ddd;'>yellow</span> suggested</div>
                        <div><span style='background:#b6f3b6;padding:2px 6px;border:1px solid #ddd;'>green</span> accepted</div>
                        <div><span style='background:#f6c1c1;padding:2px 6px;border:1px solid #ddd;'>red</span> rejected</div>
                        <div><span style='background:#e0e0e0;padding:2px 6px;border:1px solid #ddd;'>gray</span> defaulted</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with top_cols[2]:
        if phase in {"margin", "default"} and var_cur is not None and es_cur is not None:
            cur_matrix = np.vstack([var_cur, es_cur]).T
            st.markdown(
                _matrix_table_html(
                    title="VaR / ES (current)",
                    data=cur_matrix,
                    row_labels=row_labels_clients,
                    col_labels=["VaR 99", "ES 99"],
                    cell_color=None,
                    cell_note=None,
                    row_color=None,
                    max_rows=max_clients,
                    max_cols=2,
                    decimals=2,
                ),
                unsafe_allow_html=True,
            )
        elif phase in {"trades", "decision"} and var_tent is not None and es_tent is not None:
            tent_matrix = np.vstack([var_tent, es_tent]).T
            st.markdown(
                _matrix_table_html(
                    title="VaR / ES (tentative)",
                    data=tent_matrix,
                    row_labels=row_labels_clients,
                    col_labels=["VaR 99", "ES 99"],
                    cell_color=None,
                    cell_note=None,
                    row_color=None,
                    max_rows=max_clients,
                    max_cols=2,
                    decimals=2,
                ),
                unsafe_allow_html=True,
            )

    bottom_cols = st.columns([1.4, 2.6, 1.6])
    with bottom_cols[0]:
        if r_t is not None:
            max_abs = float(np.max(np.abs(r_t))) if r_t.size else 0.0

            def ret_color(i: int, j: int) -> Optional[str]:
                return _color_from_value(float(r_t[i]), max_abs, (34, 197, 94), (239, 68, 68))

            st.markdown(
                _matrix_table_html(
                    title="Returns",
                    data=r_t.reshape(-1, 1),
                    row_labels=row_labels_instruments,
                    col_labels=["Return"],
                    cell_color=ret_color,
                    cell_note=None,
                    row_color=None,
                    max_rows=max_assets,
                    max_cols=1,
                    decimals=4,
                ),
                unsafe_allow_html=True,
            )

    show_scenarios = phase in {"margin", "default", "trades", "decision"}
    var_inst = None
    es_inst = None
    if Rs is not None:
        try:
            var_inst, es_inst = _compute_var_es(Rs, alpha=0.99)
        except Exception:
            var_inst, es_inst = None, None

    with bottom_cols[1]:
        if show_scenarios:
            if Rs is None:
                st.caption("Scenario matrix not stored.")
            else:
                omega = min(Rs.shape[0], max_scenarios)
                assets = min(Rs.shape[1], max_assets)
                Rs_view = Rs[:omega, :assets].T
                row_labels = [f"inst_{i + 1}" for i in range(Rs_view.shape[0])]
                col_labels = [f"sc_{j + 1}" for j in range(Rs_view.shape[1])]
                st.markdown(
                    _matrix_table_html(
                        title="Scenarios",
                        data=Rs_view,
                        row_labels=row_labels,
                        col_labels=col_labels,
                        cell_color=None,
                        cell_note=None,
                        row_color=None,
                        max_rows=assets,
                        max_cols=omega,
                        decimals=2,
                    ),
                    unsafe_allow_html=True,
                )

    with bottom_cols[2]:
        if show_scenarios and var_inst is not None and es_inst is not None:
            inst_matrix = np.vstack([var_inst, es_inst]).T
            st.markdown(
                _matrix_table_html(
                    title="VaR / ES (per inst)",
                    data=inst_matrix,
                    row_labels=[f"inst_{i + 1}" for i in range(inst_matrix.shape[0])],
                    col_labels=["VaR 99", "ES 99"],
                    cell_color=None,
                    cell_note=None,
                    row_color=None,
                    max_rows=max_assets,
                    max_cols=2,
                    decimals=2,
                ),
                unsafe_allow_html=True,
            )

    if phase == "decision":
        qubo_delta = _to_numpy(rec.get("deltaM"))
        qubo_x = _to_numpy(rec.get("x"))
        budget = rec.get("budget_B")
        penalty = rec.get("lambda_budget")
        energy = rec.get("qubo_energy")
        qubo_num_vars = rec.get("qubo_num_vars")
        qubo_num_interactions = rec.get("qubo_num_interactions")
        qubo_solver = rec.get("qubo_solver")
        qubo_solve_time = rec.get("qubo_solve_time_s")
        accepted = int(qubo_x.sum()) if qubo_x is not None else None
        total = int(qubo_x.shape[0]) if qubo_x is not None else None
        sum_delta = float(np.dot(qubo_delta, qubo_x)) if qubo_delta is not None and qubo_x is not None else None

        st.markdown("**QUBO summary**")
        qubo_rows = []
        qubo_rows.append({"label": "Sum deltaM_i x_i", "value": _format_val(sum_delta, 2) if sum_delta is not None else "n/a"})
        qubo_rows.append({"label": "Budget B", "value": _format_val(float(budget), 2) if budget is not None else "n/a"})
        qubo_rows.append({"label": "Penalty", "value": _format_val(float(penalty), 2) if penalty is not None else "n/a"})
        qubo_rows.append({"label": "Energy", "value": _format_val(float(energy), 2) if energy is not None else "n/a"})
        if qubo_num_vars is not None or qubo_num_interactions is not None:
            size_val = f"vars={qubo_num_vars}, interactions={qubo_num_interactions}"
        else:
            size_val = "n/a"
        qubo_rows.append({"label": "QUBO size", "value": size_val})
        qubo_rows.append({"label": "Solver", "value": str(qubo_solver) if qubo_solver is not None else "n/a"})
        qubo_rows.append({"label": "Solve time", "value": f"{float(qubo_solve_time):.3f}s" if qubo_solve_time is not None else "n/a"})
        if accepted is not None and total is not None:
            qubo_rows.append({"label": "Accepted", "value": f"{accepted} / {total}"})
        qubo_df = pd.DataFrame(qubo_rows)
        st.dataframe(qubo_df, use_container_width=True, hide_index=True)

    with st.expander("Instrument view", expanded=False):
        market_df = _market_dataframe(records)
        if not market_df.empty:
            market_df = market_df[market_df["day"] <= day]
        inst_cols = [c for c in market_df.columns if c.startswith("inst_")] if not market_df.empty else []
        if not inst_cols and Rs is not None:
            inst_cols = [f"inst_{i + 1}" for i in range(Rs.shape[1])]

        if not inst_cols:
            st.caption("Instrument data not available.")
        else:
            inst_choice = st.selectbox("Instrument", inst_cols, index=0, key="instrument_view")
            view_cols = st.columns(2)
            with view_cols[0]:
                if not market_df.empty and inst_choice in market_df.columns:
                    fig, ax = plt.subplots(figsize=(4.4, 2.6))
                    ax.plot(market_df["day"], market_df[inst_choice], color="#f59e0b", marker="o", linewidth=1.5)
                    ax.set_title(f"{inst_choice} returns (to day {day})")
                    ax.set_xlabel("day")
                    ax.set_ylabel("return")
                    fig.tight_layout()
                    st.pyplot(fig)
                else:
                    st.caption("Return history not available.")
            with view_cols[1]:
                if show_scenarios and Rs is not None:
                    try:
                        idx = int(inst_choice.split("_")[1]) - 1
                    except Exception:
                        idx = inst_cols.index(inst_choice) if inst_choice in inst_cols else 0
                    if idx >= Rs.shape[1]:
                        st.caption(f"Scenario returns stored for first {Rs.shape[1]} instruments.")
                    else:
                        series = Rs[:, idx]
                        fig2, ax2 = plt.subplots(figsize=(4.4, 2.6))
                        ax2.hist(series, bins=12, color="#f59e0b", edgecolor="#92400e")
                        ax2.set_title(f"{inst_choice} scenario returns")
                        ax2.set_xlabel("return")
                        ax2.set_ylabel("count")
                        if var_inst is not None and idx < len(var_inst):
                            ax2.axvline(-var_inst[idx], color="#ef4444", linestyle="--", linewidth=1.2)
                            ax2.text(-var_inst[idx], ax2.get_ylim()[1] * 0.9, "VaR", color="#ef4444", fontsize=8)
                        fig2.tight_layout()
                        st.pyplot(fig2)
                else:
                    st.caption("Scenario distribution available in Margins, Default, Suggested Trades, and QUBO Clearing phases.")

    with st.expander("Client view", expanded=False):
        pnl_df = _pnl_dataframe(records)
        if not pnl_df.empty:
            pnl_df = pnl_df[pnl_df["day"] <= day]
        wealth_df = _wealth_dataframe(records)
        if not wealth_df.empty:
            wealth_df = wealth_df[wealth_df["day"] <= day]
        if row_labels_clients:
            cl_choice = st.selectbox("Client", row_labels_clients, index=0, key="client_view")
        else:
            cl_choice = None

        client_cols = st.columns(2)
        with client_cols[0]:
            if cl_choice is None or wealth_df.empty or cl_choice not in wealth_df.columns:
                st.caption("Wealth history not available.")
            else:
                series = wealth_df[cl_choice].to_numpy()
                valid = np.isfinite(series)
                if not np.any(valid):
                    st.caption("Wealth history not available.")
                else:
                    base = float(series[valid][0])
                    figw, axw = plt.subplots(figsize=(4.4, 2.6))
                    axw.plot(wealth_df["day"], series, color="#111827", linewidth=1.5)
                    colors = ["#22c55e" if val >= base else "#ef4444" for val in series]
                    axw.scatter(wealth_df["day"], series, c=colors, s=24, zorder=3)
                    axw.axhline(base, color="#9ca3af", linestyle="--", linewidth=1.2)
                    axw.set_title(f"{cl_choice} wealth (to day {day})")
                    axw.set_xlabel("day")
                    axw.set_ylabel("wealth")
                    figw.tight_layout()
                    st.pyplot(figw)

            if cl_choice is None or pnl_df.empty or cl_choice not in pnl_df.columns:
                st.caption("PnL history not available.")
            else:
                fig3, ax3 = plt.subplots(figsize=(4.4, 2.6))
                ax3.plot(pnl_df["day"], pnl_df[cl_choice], color="#22c55e", marker="o", linewidth=1.5)
                ax3.set_title(f"{cl_choice} PnL (to day {day})")
                ax3.set_xlabel("day")
                ax3.set_ylabel("pnl")
                fig3.tight_layout()
                st.pyplot(fig3)

        with client_cols[1]:
            if show_scenarios and Rs_port is not None and P_port is not None and cl_choice is not None:
                try:
                    cl_idx = int(cl_choice.split("_")[1]) - 1
                except Exception:
                    cl_idx = row_labels_clients.index(cl_choice) if cl_choice in row_labels_clients else 0
                losses = -(Rs_port @ P_port[cl_idx])
                fig4, ax4 = plt.subplots(figsize=(4.4, 2.6))
                ax4.hist(losses, bins=12, color="#94a3b8", edgecolor="#334155")
                ax4.set_title(f"{cl_choice} portfolio loss")
                ax4.set_xlabel("loss")
                ax4.set_ylabel("count")
                var_line = None
                if phase in {"margin", "default"} and var_cur is not None:
                    var_line = var_cur
                elif phase in {"trades", "decision"} and var_tent is not None:
                    var_line = var_tent
                elif var_cur is not None:
                    var_line = var_cur
                if var_line is not None and cl_idx < len(var_line):
                    ax4.axvline(var_line[cl_idx], color="#ef4444", linestyle="--", linewidth=1.2)
                    ax4.text(var_line[cl_idx], ax4.get_ylim()[1] * 0.9, "VaR", color="#ef4444", fontsize=8)
                fig4.tight_layout()
                st.pyplot(fig4)
                if rs_trunc_note is not None:
                    st.caption(rs_trunc_note)
            else:
                st.caption("Scenario risk available in Margins, Default, Suggested Trades, and QUBO Clearing phases.")


if __name__ == "__main__":
    main()
