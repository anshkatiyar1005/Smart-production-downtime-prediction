"""Smart Production Downtime Prediction App.

The supplied telemetry has no recorded downtime labels. This app therefore
trains an unsupervised Isolation Forest to flag unusual operating telemetry;
its risk bands are condition anomaly indicators, not downtime probabilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "synthetic_underground_mining_telemetry.csv"
SENSORS = ["temperature_c", "vibration_mm_s", "pressure_kpa", "current_a", "rpm", "uptime_percent"]
SENSOR_LABELS = {
    "temperature_c": "Temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "pressure_kpa": "Pressure (kPa)",
    "current_a": "Current (A)",
    "rpm": "Engine speed (RPM)",
    "uptime_percent": "Uptime (%)",
}
FEATURES = SENSORS + [f"delta_{col}" for col in SENSORS]
COLORS = {"Low": "#18785E", "Watch": "#D38B26", "High": "#C84E42"}

st.set_page_config(page_title="Smart Production Downtime Prediction App", page_icon="⚙️", layout="wide")


@st.cache_data(show_spinner="Training anomaly detection model on the telemetry CSV…")
def load_train_and_score(path_text: str, modified_time: float):
    """Load supplied telemetry, train an unsupervised ML model, and score samples."""
    frame = pd.read_csv(path_text)
    required = {"asset_id", "asset_name", "line", "observed_at", "operating_state", *SENSORS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    for col in SENSORS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["asset_id", "observed_at", *SENSORS]).copy()
    frame = frame.sort_values(["asset_id", "observed_at"]).reset_index(drop=True)
    # Short-term change features let the detector notice abrupt movement as well
    # as unusual absolute values. Grouping avoids differences between assets.
    for col in SENSORS:
        frame[f"delta_{col}"] = frame.groupby("asset_id")[col].diff().fillna(0)
    if len(frame) < 20:
        raise ValueError("At least 20 valid telemetry rows are needed to fit the model.")
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=250, contamination="auto", random_state=42),
    )
    model.fit(frame[FEATURES])
    # IsolationForest's decision_function is higher for normal observations.
    frame["Anomaly score"] = -model.decision_function(frame[FEATURES])
    watch_cut = float(frame["Anomaly score"].quantile(0.90))
    high_cut = float(frame["Anomaly score"].quantile(0.97))
    frame["Risk band"] = np.select(
        [frame["Anomaly score"] >= high_cut, frame["Anomaly score"] >= watch_cut],
        ["High", "Watch"],
        default="Low",
    )
    return frame, model, watch_cut, high_cut


def style_app():
    st.markdown(
        """
        <style>
        :root { --ink:#18252D; --muted:#71808A; --green:#167D64; --line:#E4E9E7; --canvas:#F3F6F5; }
        html, body, [class*="css"] { font-family:Inter,"Segoe UI",Arial,sans-serif; }
        .stApp { background:var(--canvas); }
        [data-testid="stSidebar"] { background:#17252D; border-right:1px solid #263740; }
        [data-testid="stSidebar"] * { color:#E9F0EE; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.28rem .15rem; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#A8B7B8; }
        .block-container { max-width:1480px; padding:2.25rem 2.5rem 3rem; }
        .eyebrow { color:var(--green); font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
        .title { color:var(--ink); font-size:2.15rem; line-height:1.18; font-weight:700; letter-spacing:-.045em; margin:.25rem 0 .45rem; }
        .subtitle { color:var(--muted); font-size:.96rem; margin:0 0 1.15rem; }
        .card { background:#FFF; border:1px solid var(--line); border-top:3px solid var(--accent,#167D64); border-radius:11px; padding:15px 18px 14px; min-height:112px; box-shadow:0 2px 8px rgba(24,37,45,.035); transition:box-shadow .15s ease,transform .15s ease; }
        .card:hover { box-shadow:0 6px 18px rgba(24,37,45,.08); transform:translateY(-1px); }
        .label { color:var(--muted); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.075em; }
        .value { color:var(--ink); font-size:1.85rem; font-weight:700; line-height:1.2; letter-spacing:-.045em; margin:.45rem 0 .15rem; }
        .note { color:var(--muted); font-size:.77rem; line-height:1.4; }
        .panel-title { color:var(--ink); font-size:1.02rem; font-weight:700; margin:.35rem 0 .2rem; }
        .panel-note { color:var(--muted); font-size:.8rem; margin-bottom:.8rem; }
        .notice { padding:12px 15px; border-radius:8px; background:#FFF8EA; color:#70551E; border:1px solid #EEDDBA; border-left:4px solid #D89B32; font-size:.83rem; line-height:1.55; margin:10px 0 18px; }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] { background:#E7F1ED; color:#145E4C; border-radius:6px; }
        [data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div { border-color:#DCE4E0; border-radius:8px; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:9px; overflow:hidden; }
        [data-testid="stPlotlyChart"] { background:#FFF; border:1px solid var(--line); border-radius:10px; padding:6px 8px 0; }
        hr { border-color:rgba(225,235,231,.2); }
        @media (max-width:800px) { .block-container { padding:1.3rem 1rem 2rem; } .title { font-size:1.75rem; } }
        </style>
        """, unsafe_allow_html=True,
    )


def metric(label, value, note, accent="#167D64"):
    st.markdown(f'<div class="card" style="--accent:{accent}"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def main():
    style_app()
    with st.sidebar:
        st.markdown("<div style='font-size:1.28rem;font-weight:750;letter-spacing:.04em'>⚙ FORGE <span style='color:#73C4A4'>/</span></div>", unsafe_allow_html=True)
        st.caption("OPERATIONS INTELLIGENCE")
        st.divider()
        page = st.radio("WORKSPACE", ["Operations overview", "Asset explorer", "Model & data notes"])
        st.divider()
        st.markdown("**MODEL**")
        st.caption("Isolation Forest · unsupervised anomaly detection")
        st.markdown("**DATASET**")
        st.caption("Synthetic underground mining telemetry")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Keep the `data` folder beside `app.py`.")
        st.stop()
    try:
        data, model, watch_cut, high_cut = load_train_and_score(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not train the model: {exc}")
        st.stop()

    latest = data.sort_values("observed_at").groupby("asset_id", as_index=False).tail(1).copy()
    latest = latest.sort_values("Anomaly score", ascending=False)
    assets = sorted(latest["asset_id"].unique())
    if page == "Model & data notes":
        st.markdown('<div class="eyebrow">MODEL CARD</div><div class="title">How this demo works</div>', unsafe_allow_html=True)
        st.markdown('<div class="notice"><b>Downtime labels are not present.</b> Every source record is marked RUNNING. This app trains an unsupervised anomaly detector on the supplied sensor history. Its risk bands show unusual telemetry relative to this dataset; they are not probabilities that downtime will occur.</div>', unsafe_allow_html=True)
        st.markdown("### Training details")
        st.write(f"The app trains a scikit-learn Isolation Forest on {len(data):,} telemetry rows from {data['asset_id'].nunique()} assets. Inputs include temperature, vibration, pressure, current, RPM, uptime, and within-asset sensor changes between readings.")
        st.write("The Watch and High bands mark the top 10% and top 3% of anomaly scores in the training data. This is a relative screening rule, not a validated maintenance threshold.")
        st.markdown("### Dataset quality summary")
        st.write(f"Time range: {data['observed_at'].min():%Y-%m-%d %H:%M UTC} to {data['observed_at'].max():%Y-%m-%d %H:%M UTC}. Recorded operating states: {', '.join(sorted(data['operating_state'].astype(str).unique()))}.")
        st.write("The telemetry is synthetic, with five assets and fewer than one month of observations. Validate against real labeled downtime events before describing outputs as downtime predictions.")
        st.markdown("### Threshold comparison")
        thresholds = []
        for label, signal, threshold_col, comparator in [
            ("Temperature warning", "temperature_c", "temperature_warning_c", "above"),
            ("Vibration warning", "vibration_mm_s", "vibration_warning_mm_s", "above"),
            ("Pressure low warning", "pressure_kpa", "pressure_warning_kpa", "below"),
            ("Pressure high warning", "pressure_kpa", "pressure_high_warning_kpa", "above"),
        ]:
            if threshold_col in data.columns:
                breached = data[signal] <= data[threshold_col] if comparator == "below" else data[signal] >= data[threshold_col]
                thresholds.append({"Check": label, "Threshold breaches": int(breached.sum())})
        st.dataframe(pd.DataFrame(thresholds), hide_index=True, use_container_width=True)
        return

    st.markdown('<div class="eyebrow">PLANT OPERATIONS / RELIABILITY</div><div class="title">Smart Production Downtime Prediction App</div><div class="subtitle">Machine learning scans asset telemetry for unusual operating conditions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Interpretation:</b> This dataset has no downtime events. The model flags telemetry anomalies as an early inspection signal; it does not estimate downtime probability.</div>', unsafe_allow_html=True)
    selected = st.multiselect("Filter assets", assets, default=assets, label_visibility="collapsed", placeholder="Filter assets")
    current = latest[latest["asset_id"].isin(selected)]
    high_n = int((current["Risk band"] == "High").sum())
    watch_n = int((current["Risk band"] == "Watch").sum())
    r1, r2, r3, r4 = st.columns(4)
    with r1: metric("Assets monitored", str(len(current)), "Latest reading per selected asset", "#167D64")
    with r2: metric("High anomaly", str(high_n), "Top 3% score cutoff on training set", "#C84E42")
    with r3: metric("Watch list", str(watch_n), "Top 10% score cutoff on training set", "#D38B26")
    with r4: metric("Training records", f"{len(data):,}", f"{data['asset_id'].nunique()} assets · model fitted in app", "#5674A5")

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Sensor history</div><div class="panel-note">Choose a sensor to inspect readings over time</div>', unsafe_allow_html=True)
        sensor = st.selectbox("Sensor", SENSORS, format_func=lambda x: SENSOR_LABELS[x], label_visibility="collapsed")
        history = data[data["asset_id"].isin(selected)]
        chart = px.line(history, x="observed_at", y=sensor, color="asset_name", labels={"observed_at":"Observed at", sensor:SENSOR_LABELS[sensor], "asset_name":"Asset"}, color_discrete_sequence=["#167D64", "#D38B26", "#637BC4", "#C84E42", "#8A6DB1"])
        chart.update_layout(height=320, margin=dict(l=0,r=0,t=5,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#77817C", size=11), legend=dict(orientation="h", y=1.12, x=0), xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#E8EBE5", zeroline=False))
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar":False})
    with right:
        st.markdown('<div class="panel-title">Current asset condition</div><div class="panel-note">Latest ML anomaly band for each selected asset</div>', unsafe_allow_html=True)
        mix = current["Risk band"].value_counts().reindex(["High", "Watch", "Low"], fill_value=0).rename_axis("Risk band").reset_index(name="Assets")
        donut = px.pie(mix, values="Assets", names="Risk band", hole=.72, color="Risk band", color_discrete_map=COLORS)
        donut.update_traces(textinfo="none", marker=dict(line=dict(color="white", width=3)))
        donut.update_layout(height=320, margin=dict(l=0,r=0,t=5,b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#77817C"), legend=dict(orientation="h", y=-.02, x=.0), annotations=[dict(text=f"{len(current)}<br><span style='font-size:11px;color:#77817C'>assets</span>", x=.5,y=.5,showarrow=False,font=dict(size=24,color="#17211F"))])
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="panel-title">Asset watchlist</div><div class="panel-note">Sorted by anomaly score · high scores are unusual relative to the supplied synthetic baseline</div>', unsafe_allow_html=True)
    display = current[["asset_id", "asset_name", "line", "observed_at", "operating_state", "Risk band", "Anomaly score", *SENSORS]].copy()
    display["observed_at"] = display["observed_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(display, use_container_width=True, hide_index=True, height=260, column_config={"asset_id":"Asset ID", "asset_name":"Equipment", "line":"Production line", "observed_at":"Latest reading", "operating_state":"Recorded state", "Risk band":st.column_config.TextColumn("Anomaly band"), "Anomaly score":st.column_config.NumberColumn("Anomaly score", format="%.4f")})

    if page == "Asset explorer" and not current.empty:
        st.markdown('<div class="panel-title">Asset explorer</div>', unsafe_allow_html=True)
        chosen = st.selectbox("Choose equipment", current["asset_id"].tolist(), format_func=lambda x: f"{x} · {current.loc[current.asset_id == x, 'asset_name'].iloc[0]}")
        asset_history = data[data["asset_id"] == chosen]
        latest_row = asset_history.iloc[-1]
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**{latest_row['asset_name']}** · {latest_row['line']}")
            st.write(f"Latest operating state: **{latest_row['operating_state']}**")
            st.write(f"Latest anomaly band: **{latest_row['Risk band']}**")
            st.write("Suggested action: inspect the asset and review sensor trends if its anomaly score is elevated.")
        with c2:
            values = pd.DataFrame({"Sensor": [SENSOR_LABELS[c] for c in SENSORS], "Latest reading": [latest_row[c] for c in SENSORS]})
            st.dataframe(values, hide_index=True, use_container_width=True)

    st.caption("Model: scikit-learn Isolation Forest trained at app startup on the included telemetry CSV. Risk bands are unsupervised anomaly rankings, not downtime probabilities.")


if __name__ == "__main__":
    main()
