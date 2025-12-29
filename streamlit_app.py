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

PHASE_ORDER = ["start", "market", "margin", "trades", "decision", "end"]
PHASE_LABELS = {
    "start": "Start",
    "market": "Market",
    "margin": "Pre-Trade",
    "trades": "Trades",
    "decision": "Decision",
    "end": "End",
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
        if rec.get("phase") != "market":
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
        return _color_from_value(
            float(data[i, j]),
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
        return f"{sign}{dp:.2f}"

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

    records = _torch_load(path)
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
    var_cur = None
    es_cur = None
    var_tent = None
    es_tent = None
    if Rs is not None and P is not None:
        try:
            var_cur, es_cur = _portfolio_var_es(P, Rs, alpha=0.99)
            if phase in {"trades", "decision"} and DeltaP is not None:
                var_tent, es_tent = _portfolio_var_es(P + DeltaP, Rs, alpha=0.99)
        except Exception:
            var_cur, es_cur, var_tent, es_tent = None, None, None, None

    if alive is None and W is not None:
        alive = np.ones(W.shape[0], dtype=bool)

    if z_t is not None:
        st.caption(f"Market state z_t: {z_t}")

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
        stats_cols = ["Alive", "W", "C", "M", "PnL"]
        stats_data = []
        if alive is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(alive.astype(float))
        if W is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(W)
        if C is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(C)
        if M_req_cur is not None:
            stats_data.append(M_req_cur)
        elif M_req_tent is not None:
            stats_data.append(M_req_tent)
        else:
            stats_data.append(np.full((client_count or 0,), np.nan))
        if pnl is None:
            stats_data.append(np.full((client_count or 0,), np.nan))
        else:
            stats_data.append(pnl)
        if phase == "trades":
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
            if "DeltaM" in stats_cols:
                idx = stats_cols.index("DeltaM")
                dm_col = stats_matrix[:, idx]
                max_abs = float(np.nanmax(np.abs(dm_col))) if dm_col.size and not np.all(np.isnan(dm_col)) else 0.0
                color_rules["DeltaM"] = {"pos_rgb": (249, 115, 22), "neg_rgb": (59, 130, 246), "max_abs": max_abs}
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
        if phase == "margin" and var_cur is not None and es_cur is not None:
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

    show_scenarios = phase in {"margin", "trades", "decision"}

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
        if show_scenarios and Rs is not None:
            var_inst, es_inst = _compute_var_es(Rs, alpha=0.99)
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

    with st.expander("Scenario distributions", expanded=False):
        if show_scenarios:
            if Rs is None:
                st.caption("Scenarios not available.")
            else:
                hist_cols = st.columns(2)
                inst_labels = [f"inst_{i + 1}" for i in range(Rs.shape[1])]
                with hist_cols[0]:
                    inst_choice = st.selectbox("Instrument", inst_labels, index=0, key="inst_hist")
                    idx = inst_labels.index(inst_choice)
                    series = Rs[:, idx]
                    fig, ax = plt.subplots(figsize=(3.6, 2.4))
                    ax.hist(series, bins=12, color="#38bdf8", edgecolor="#1e3a8a")
                    ax.set_title(f"{inst_choice} returns")
                    ax.set_xlabel("Return")
                    ax.set_ylabel("Count")
                    fig.tight_layout()
                    st.pyplot(fig)
                with hist_cols[1]:
                    if P is not None and row_labels_clients:
                        cl_labels = row_labels_clients
                        cl_choice = st.selectbox("Client", cl_labels, index=0, key="client_hist")
                        cl_idx = cl_labels.index(cl_choice)
                        losses = -(Rs @ P[cl_idx])
                        fig2, ax2 = plt.subplots(figsize=(3.6, 2.4))
                        ax2.hist(losses, bins=12, color="#94a3b8", edgecolor="#334155")
                        ax2.set_title(f"{cl_choice} loss")
                        ax2.set_xlabel("Loss")
                        ax2.set_ylabel("Count")
                        if var_cur is not None and cl_idx < len(var_cur):
                            ax2.axvline(var_cur[cl_idx], color="#ef4444", linestyle="--", linewidth=1)
                            ax2.text(var_cur[cl_idx], ax2.get_ylim()[1] * 0.9, "VaR", color="#ef4444", fontsize=8)
                        fig2.tight_layout()
                        st.pyplot(fig2)
                    else:
                        st.caption("Client loss histogram not available.")
        else:
            st.caption("Scenario views are available in Pre-Trade, Trades, and Decision phases.")

    if phase == "decision":
        qubo_delta = _to_numpy(rec.get("deltaM"))
        qubo_x = _to_numpy(rec.get("x"))
        budget = rec.get("budget_B")
        penalty = rec.get("lambda_budget")
        energy = rec.get("qubo_energy")
        accepted = int(qubo_x.sum()) if qubo_x is not None else None
        total = int(qubo_x.shape[0]) if qubo_x is not None else None
        sum_delta = float(np.dot(qubo_delta, qubo_x)) if qubo_delta is not None and qubo_x is not None else None

        st.markdown("**QUBO summary**")
        qubo_rows = []
        qubo_rows.append({"label": "Σ ΔM_i x_i", "value": _format_val(sum_delta, 2) if sum_delta is not None else "n/a"})
        qubo_rows.append({"label": "Budget B", "value": _format_val(float(budget), 2) if budget is not None else "n/a"})
        qubo_rows.append({"label": "Penalty", "value": _format_val(float(penalty), 2) if penalty is not None else "n/a"})
        qubo_rows.append({"label": "Energy", "value": _format_val(float(energy), 2) if energy is not None else "n/a"})
        if accepted is not None and total is not None:
            qubo_rows.append({"label": "Accepted", "value": f"{accepted} / {total}"})
        qubo_df = pd.DataFrame(qubo_rows)
        st.dataframe(qubo_df, use_container_width=True, hide_index=True)

    with st.expander("Returns over days", expanded=False):
        market_df = _market_dataframe(records)
        if not market_df.empty:
            market_df = market_df[market_df["day"] <= day]
            if not market_df.empty:
                st.line_chart(market_df.set_index("day")[[c for c in market_df.columns if c.startswith("inst_")]])
                st.line_chart(market_df.set_index("day")[["z_t"]])
        else:
            st.info("No market phase records available.")


if __name__ == "__main__":
    main()
