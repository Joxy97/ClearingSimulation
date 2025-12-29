from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

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
                    c_style = f" style='background-color:{color};'"
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
) -> str:
    def row_color(i: int) -> Optional[str]:
        if alive is None:
            return None
        return "#e8e8e8" if not bool(alive[i]) else None

    def cell_color(i: int, j: int) -> Optional[str]:
        if alive is None:
            return None
        return "#e8e8e8" if not bool(alive[i]) else None

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

    row_labels = [f"c{i + 1}" for i in range(P.shape[0])]
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


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #f8fafc; }
        .tensor-title { font-weight: 700; margin-bottom: 6px; color: #1f2937; }
        .tensor-wrap { overflow-x: auto; }
        table.tensor-table { border-collapse: collapse; width: 100%; background: #fff; }
        table.tensor-table th, table.tensor-table td {
            border: 1px solid #cfcfcf; padding: 6px 8px; text-align: center; font-size: 12px; color: #111827;
            width: 56px; height: 36px;
        }
        table.tensor-table { table-layout: fixed; }
        table.tensor-table th { background: #f3f4f6; font-weight: 600; color: #1f2937; }
        .cell-val { font-weight: 600; font-size: 12px; color: #111827; }
        .cell-note { font-size: 10px; color: #4b5563; }
        .section-label { font-weight: 700; color: #1f2937; margin-bottom: 6px; }
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
        row_labels_clients = [f"{i + 1}" for i in range(client_count)]

    row_labels_instruments = []
    inst_count = None
    for vec in (r_t,):
        if vec is not None:
            inst_count = vec.shape[0]
            break
    if inst_count is not None:
        row_labels_instruments = [f"inst_{i + 1}" for i in range(inst_count)]

    top_cols = st.columns([1, 2, 4])
    with top_cols[0]:
        if alive is not None:
            alive_vec = alive.astype(int)
            st.markdown(
                _vector_table_html(
                    title="Client",
                    data=alive_vec,
                    row_labels=row_labels_clients,
                    max_rows=max_clients,
                    col_label="Alive",
                    decimals=0,
                ),
                unsafe_allow_html=True,
            )

    with top_cols[1]:
        stats_cols = ["W", "C", "M", "PnL"]
        stats_data = []
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

        if stats_data:
            stats_matrix = np.vstack(stats_data).T
            st.markdown(
                _stats_table_html(
                    title="Stats",
                    data=stats_matrix,
                    row_labels=row_labels_clients,
                    col_labels=stats_cols,
                    alive=alive,
                    max_rows=max_clients,
                ),
                unsafe_allow_html=True,
            )

    with top_cols[2]:
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

    bottom_cols = st.columns([1.4, 2.4, 1.2, 1.2, 2.0])
    with bottom_cols[0]:
        if r_t is not None:
            st.markdown(
                _vector_table_html(
                    title="Returns",
                    data=r_t,
                    row_labels=row_labels_instruments,
                    max_rows=max_assets,
                    col_label="Return",
                ),
                unsafe_allow_html=True,
            )

    Rs = _to_numpy(rec.get("R_scenarios"))
    var_vec = None
    es_vec = None
    if Rs is not None:
        try:
            var_vec, es_vec = _compute_var_es(Rs, alpha=0.99)
        except Exception:
            var_vec, es_vec = None, None

    with bottom_cols[1]:
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
        if var_vec is not None:
            st.markdown(
                _vector_table_html(
                    title="Value-at-Risk",
                    data=var_vec[: max_assets],
                    row_labels=[f"inst_{i + 1}" for i in range(min(len(var_vec), max_assets))],
                    max_rows=max_assets,
                    col_label="VaR 99",
                ),
                unsafe_allow_html=True,
            )

    with bottom_cols[3]:
        if es_vec is not None:
            st.markdown(
                _vector_table_html(
                    title="ES at 99%",
                    data=es_vec[: max_assets],
                    row_labels=[f"inst_{i + 1}" for i in range(min(len(es_vec), max_assets))],
                    max_rows=max_assets,
                    col_label="ES 99",
                ),
                unsafe_allow_html=True,
            )

    with bottom_cols[4]:
        st.markdown("<div class='tensor-title'>Per instrument distribution</div>", unsafe_allow_html=True)
        if Rs is None:
            st.caption("Scenarios not available.")
        else:
            inst_labels = [f"inst_{i + 1}" for i in range(Rs.shape[1])]
            inst_choice = st.selectbox("Instrument", inst_labels, index=0)
            idx = inst_labels.index(inst_choice)
            series = Rs[:, idx]
            fig, ax = plt.subplots(figsize=(3.4, 2.2))
            ax.hist(series, bins=12, color="#38bdf8", edgecolor="#1e3a8a")
            ax.set_title(inst_choice)
            ax.set_xlabel("Return")
            ax.set_ylabel("Count")
            if var_vec is not None and idx < len(var_vec):
                ax.axvline(-var_vec[idx], color="#ef4444", linestyle="--", linewidth=1)
                ax.text(-var_vec[idx], ax.get_ylim()[1] * 0.9, "VaR", color="#ef4444", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

    with st.expander("Returns over days", expanded=False):
        market_df = _market_dataframe(records)
        if not market_df.empty:
            st.line_chart(market_df.set_index("day")[[c for c in market_df.columns if c.startswith("inst_")]])
            st.line_chart(market_df.set_index("day")[["z_t"]])
        else:
            st.info("No market phase records available.")


if __name__ == "__main__":
    main()
