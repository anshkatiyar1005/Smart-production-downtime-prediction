"""Smart Production Downtime Prediction App.

The supplied telemetry has no recorded downtime labels. This app therefore
trains an unsupervised Isolation Forest to flag unusual operating telemetry;
its risk bands are condition anomaly indicators, not downtime probabilities.
"""

from pathlib import Path
from io import BytesIO
import re

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
REQUIRED_COLUMNS = ["asset_id", "observed_at", *SENSORS]
OPTIONAL_COLUMNS = ["asset_name", "line", "operating_state"]
COLUMN_TITLES = {
    "asset_id": "Asset ID",
    "observed_at": "Timestamp",
    "temperature_c": "Temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "pressure_kpa": "Pressure (kPa)",
    "current_a": "Current (A)",
    "rpm": "Engine speed (RPM)",
    "uptime_percent": "Uptime (%)",
}
COLUMN_ALIASES = {
    "asset_id": {"asset", "machine_id", "equipment_id", "machine", "equipment"},
    "observed_at": {"timestamp", "datetime", "date_time", "time", "date"},
    "temperature_c": {"temperature", "temp", "temp_c", "temperature_c"},
    "vibration_mm_s": {"vibration", "vibration_mm_s", "vibration_mm_sec"},
    "pressure_kpa": {"pressure", "pressure_kpa"},
    "current_a": {"current", "amps", "current_a", "current_amps"},
    "rpm": {"rpm", "engine_rpm", "speed_rpm"},
    "uptime_percent": {"uptime", "uptime_pct", "uptime_percent", "availability_percent"},
}
COLORS = {"Low": "#18785E", "Watch": "#D38B26", "High": "#C84E42"}

st.set_page_config(page_title="Smart Production Downtime Prediction App", page_icon="⚙️", layout="wide")


@st.cache_data(show_spinner="Training anomaly detection model on your telemetry…")
def load_train_and_score(csv_bytes: bytes, column_mapping: tuple[tuple[str, str], ...]):
    """Normalize a telemetry CSV, fit an unsupervised model, and score its rows."""
    frame = pd.read_csv(BytesIO(csv_bytes), low_memory=False)
    mapping = dict(column_mapping)
    rename = {source: target for target, source in mapping.items() if source != target}
    frame = frame.rename(columns=rename)
    required = set(REQUIRED_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    frame["asset_id"] = frame["asset_id"].astype("string").str.strip()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    for col in SENSORS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["asset_id", "observed_at", *SENSORS]).copy()
    frame = frame[frame["asset_id"] != ""].copy()
    if "asset_name" not in frame:
        frame["asset_name"] = frame["asset_id"]
    if "line" not in frame:
        frame["line"] = "Unassigned"
    if "operating_state" not in frame:
        frame["operating_state"] = "Not supplied"
    for col in OPTIONAL_COLUMNS:
        frame[col] = frame[col].fillna("Not supplied").astype(str)
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


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def suggest_column(target: str, headers: list[str]) -> str | None:
    normalized = {normalize_header(header): header for header in headers}
    if target in normalized:
        return normalized[target]
    for alias in COLUMN_ALIASES.get(target, set()):
        if alias in normalized:
            return normalized[alias]
    return None


def csv_template() -> bytes:
    columns = [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
    return pd.DataFrame(columns=columns).to_csv(index=False).encode("utf-8")


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


def sensor_chart(history: pd.DataFrame, sensor: str, show_legend: bool = True, height: int = 380):
    """Draw a readable sensor trend with compact labels and a reserved legend row."""
    chart = px.line(
        history,
        x="observed_at",
        y=sensor,
        color="asset_id",
        hover_data={"asset_name": True, "asset_id": True, "observed_at": True},
        labels={
            "observed_at": "",
            sensor: SENSOR_LABELS[sensor],
            "asset_id": "Asset ID",
            "asset_name": "Equipment",
        },
        color_discrete_sequence=["#147A63", "#D08A25", "#5374B5", "#C4554D", "#8269AE"],
    )
    chart.update_traces(line=dict(width=2.35), connectgaps=False)
    chart.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=12, b=88 if show_legend else 58),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#66757D", size=11),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            x=0,
            y=-0.26,
            xanchor="left",
            yanchor="top",
            title=None,
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=10, color="#52616A"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(title=None, showgrid=False, tickformat="%b %d", tickfont=dict(size=10), automargin=True),
        yaxis=dict(title=SENSOR_LABELS[sensor], gridcolor="#E7ECEA", zeroline=False, tickfont=dict(size=10), automargin=True),
        hoverlabel=dict(font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12)),
    )
    return chart


def main():
    style_app()
    with st.sidebar:
        st.markdown("<div style='font-size:1.28rem;font-weight:750;letter-spacing:.04em'>⚙ FORGE <span style='color:#73C4A4'>/</span></div>", unsafe_allow_html=True)
        st.caption("OPERATIONS INTELLIGENCE")
        st.divider()
        page = st.radio("WORKSPACE", ["Operations overview", "Asset detail", "Model & data notes"])
        st.divider()
        st.markdown("**MODEL**")
        st.caption("Isolation Forest · unsupervised anomaly detection")
        st.divider()
        st.markdown("**TRAIN WITH YOUR DATA**")
        uploaded_file = st.file_uploader(
            "Upload a telemetry CSV",
            type=["csv"],
            help="Required: asset ID, timestamp, and the six sensor readings. You can map differently named columns after uploading.",
        )
        st.download_button(
            "Download CSV template",
            data=csv_template(),
            file_name="telemetry_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Use °C, mm/s, kPa, A, RPM and percent. Include at least 20 valid rows. Optional: asset name, line and operating state.")
        st.caption("Uploads are processed by this hosted app and are not added to the GitHub repository.")

        uploaded_bytes = None
        source_name = "Synthetic underground mining telemetry"
        is_uploaded = uploaded_file is not None
        mapping = tuple((column, column) for column in REQUIRED_COLUMNS)
        if is_uploaded:
            uploaded_bytes = uploaded_file.getvalue()
            source_name = uploaded_file.name
            try:
                headers = [str(col) for col in pd.read_csv(BytesIO(uploaded_bytes), nrows=0).columns]
            except Exception as exc:
                st.error(f"Could not read the uploaded CSV header: {exc}")
                st.stop()

            guesses = {target: suggest_column(target, headers) for target in REQUIRED_COLUMNS}
            needs_mapping = any(value is None for value in guesses.values())
            if needs_mapping:
                with st.expander("Map your CSV columns", expanded=True):
                    st.caption("Choose the source column for each required model input.")
                    options = ["— Select a column —", *headers]
                    choices = {}
                    for target in REQUIRED_COLUMNS:
                        guess = guesses[target]
                        default_index = options.index(guess) if guess in headers else 0
                        choices[target] = st.selectbox(
                            COLUMN_TITLES[target],
                            options,
                            index=default_index,
                            key=f"column-map-{target}",
                        )
                    mapping = tuple((target, source) for target, source in choices.items() if source != options[0])
                    if len(mapping) != len(REQUIRED_COLUMNS):
                        st.warning("Map all required columns to train the model.")
                        st.stop()
                    if len({source for _, source in mapping}) != len(REQUIRED_COLUMNS):
                        st.warning("Each required input must map to a different CSV column.")
                        st.stop()
            else:
                mapping = tuple((target, guesses[target]) for target in REQUIRED_COLUMNS)
                st.caption("Required columns detected automatically.")

    if is_uploaded:
        csv_bytes = uploaded_bytes
    elif DATA_PATH.exists():
        csv_bytes = DATA_PATH.read_bytes()
    else:
        st.error(f"Dataset not found at `{DATA_PATH}`. Keep the `data` folder beside `app.py`.")
        st.stop()
    try:
        data, model, watch_cut, high_cut = load_train_and_score(csv_bytes, mapping)
    except Exception as exc:
        st.error(f"Could not train the model: {exc}")
        st.stop()
    source_description = f"Uploaded dataset: {source_name}" if is_uploaded else "Demo dataset: synthetic underground mining telemetry"

    latest = data.sort_values("observed_at").groupby("asset_id", as_index=False).tail(1).copy()
    latest = latest.sort_values("Anomaly score", ascending=False)
    assets = sorted(latest["asset_id"].unique())
    if page == "Model & data notes":
        st.markdown('<div class="eyebrow">MODEL CARD</div><div class="title">Model & data notes</div>', unsafe_allow_html=True)
        st.info(source_description)
        st.markdown('<div class="notice"><b>Model scope:</b> The Isolation Forest trains on sensor readings and their within-asset changes. It is unsupervised and does not use downtime labels. Risk bands identify unusual readings relative to the selected dataset; they are not probabilities that downtime will occur.</div>', unsafe_allow_html=True)
        st.markdown("### Training details")
        st.write(f"The app trains a scikit-learn Isolation Forest on {len(data):,} telemetry rows from {data['asset_id'].nunique()} assets. Inputs include temperature, vibration, pressure, current, RPM, uptime, and within-asset sensor changes between readings.")
        st.write("The Watch and High bands mark the top 10% and top 3% of anomaly scores in the training data. This is a relative screening rule, not a validated maintenance threshold.")
        st.markdown("### Dataset quality summary")
        st.write(f"Time range: {data['observed_at'].min():%Y-%m-%d %H:%M UTC} to {data['observed_at'].max():%Y-%m-%d %H:%M UTC}. Recorded operating states: {', '.join(sorted(data['operating_state'].astype(str).unique()))}.")
        if is_uploaded:
            st.write("The uploaded dataset is used to retrain the anomaly detector for this session. The operating-state column, if provided, is displayed as context and is not treated as a downtime target.")
        else:
            st.write("The built-in telemetry is synthetic, with five assets and fewer than one month of observations. Validate against real labeled downtime events before describing outputs as downtime predictions.")
        st.markdown("### CSV format")
        st.write("Download the template from the sidebar. Required fields are asset ID, timestamp, and all six sensor columns. The optional fields `asset_name`, `line`, and `operating_state` add context to the dashboard.")
        with st.expander("Required column names"):
            st.code(", ".join(REQUIRED_COLUMNS), language="text")
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
        if thresholds:
            st.dataframe(pd.DataFrame(thresholds), hide_index=True, use_container_width=True)
        else:
            st.caption("No sensor warning-threshold columns were included in this dataset.")
        return

    if page == "Asset detail":
        st.markdown('<div class="eyebrow">ASSET INTELLIGENCE</div><div class="title">Asset condition detail</div><div class="subtitle">Review the latest condition and sensor history for an individual machine.</div>', unsafe_allow_html=True)
        chosen = st.selectbox("Select asset", assets, format_func=lambda x: f"{x} · {latest.loc[latest.asset_id == x, 'asset_name'].iloc[0]}")
        asset_history = data[data["asset_id"] == chosen].sort_values("observed_at")
        asset = asset_history.iloc[-1]
        a1, a2, a3, a4 = st.columns(4)
        with a1: metric("Anomaly band", asset["Risk band"], "Relative to supplied telemetry", COLORS[asset["Risk band"]])
        with a2: metric("Anomaly score", f"{asset['Anomaly score']:.3f}", "Higher values are more unusual", "#5674A5")
        with a3: metric("Operating state", str(asset["operating_state"]).title(), "Recorded in latest row", "#167D64")
        with a4: metric("Readings available", f"{len(asset_history):,}", "Six-hour sensor history", "#73818A")

        left, right = st.columns([1.3, 1], gap="large")
        with left:
            st.markdown('<div class="panel-title">Sensor trend</div><div class="panel-note">Inspect movement over the available history</div>', unsafe_allow_html=True)
            sensor = st.selectbox("Sensor", SENSORS, format_func=lambda x: SENSOR_LABELS[x], key="asset_sensor")
            st.plotly_chart(sensor_chart(asset_history, sensor, show_legend=False, height=350), use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown('<div class="panel-title">Latest readings</div><div class="panel-note">Most recent telemetry values</div>', unsafe_allow_html=True)
            values = pd.DataFrame({"Sensor": [SENSOR_LABELS[c] for c in SENSORS], "Reading": [asset[c] for c in SENSORS]})
            st.dataframe(values, hide_index=True, use_container_width=True, height=285, column_config={"Reading": st.column_config.NumberColumn("Reading", format="%.2f")})

        st.markdown('<div class="panel-title">Recent history</div><div class="panel-note">Latest 12 samples for the selected asset</div>', unsafe_allow_html=True)
        recent = asset_history.tail(12)[["observed_at", "operating_state", "Risk band", "Anomaly score", *SENSORS]].copy()
        recent["observed_at"] = recent["observed_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(recent.sort_values("observed_at", ascending=False), hide_index=True, use_container_width=True, height=390, column_config={"observed_at": "Observed at", "operating_state": "State", "Risk band": "Anomaly band", "Anomaly score": st.column_config.NumberColumn("Score", format="%.4f")})
        st.caption("Anomaly bands flag unusual telemetry and are not downtime probabilities.")
        return

    st.markdown('<div class="eyebrow">PLANT OPERATIONS / RELIABILITY</div><div class="title">Smart Production Downtime Prediction App</div><div class="subtitle">A clear view of equipment condition across the fleet.</div>', unsafe_allow_html=True)
    if is_uploaded:
        st.success(f"Custom dataset active: {source_name} · model trained on {len(data):,} valid readings across {data['asset_id'].nunique()} assets.")
    else:
        st.info("Demo mode · using the included synthetic telemetry CSV. Upload your own data from the sidebar to retrain.")
    st.markdown('<div class="notice"><b>Model scope:</b> The anomaly model ranks unusual sensor patterns for inspection; it does not estimate downtime probability.</div>', unsafe_allow_html=True)
    selected = st.multiselect("Filter assets", assets, default=assets, label_visibility="collapsed", placeholder="Filter assets")
    if not selected:
        st.info("Select at least one asset to display fleet condition.")
        st.stop()
    current = latest[latest["asset_id"].isin(selected)]
    high_n = int((current["Risk band"] == "High").sum())
    watch_n = int((current["Risk band"] == "Watch").sum())
    r1, r2, r3, r4 = st.columns(4)
    with r1: metric("Assets monitored", str(len(current)), "Latest reading per asset", "#167D64")
    with r2: metric("High anomaly", str(high_n), "Highest relative scores", "#C84E42")
    with r3: metric("Watch list", str(watch_n), "Elevated relative scores", "#D38B26")
    with r4: metric("Training records", f"{len(data):,}", f"{data['asset_id'].nunique()} assets", "#5674A5")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">Sensor history</div><div class="panel-note">Compare asset readings across the selected period</div>', unsafe_allow_html=True)
        sensor = st.selectbox("Sensor", SENSORS, format_func=lambda x: SENSOR_LABELS[x], label_visibility="collapsed")
        history = data[data["asset_id"].isin(selected)]
        st.plotly_chart(sensor_chart(history, sensor, show_legend=True, height=385), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="panel-title">Fleet condition</div><div class="panel-note">Latest anomaly band across selected assets</div>', unsafe_allow_html=True)
        mix = current["Risk band"].value_counts().reindex(["High", "Watch", "Low"], fill_value=0).rename_axis("Risk band").reset_index(name="Assets")
        donut = px.pie(mix, values="Assets", names="Risk band", hole=.72, color="Risk band", color_discrete_map=COLORS)
        donut.update_traces(textinfo="none", marker=dict(line=dict(color="white", width=3)))
        donut.update_layout(height=385, margin=dict(l=8, r=8, t=12, b=60), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#66757D", size=11), legend=dict(orientation="h", y=-.10, x=.5, xanchor="center", yanchor="top", font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=11)), annotations=[dict(text=f"{len(current)}<br><span style='font-size:11px;color:#71808A'>assets</span>", x=.5, y=.5, showarrow=False, font=dict(size=24, color="#18252D", family="Inter, Segoe UI, Arial, sans-serif"))])
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="panel-title">Maintenance watchlist</div><div class="panel-note">Latest reading per asset, ranked by anomaly score</div>', unsafe_allow_html=True)
    display = current[["asset_id", "asset_name", "line", "operating_state", "observed_at", "Risk band", "Anomaly score"]].copy()
    display["observed_at"] = display["observed_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(display, use_container_width=True, hide_index=True, height=min(360, 46 + 38 * max(len(display), 1)), column_config={"asset_id": "Asset ID", "asset_name": "Equipment", "line": "Production line", "operating_state": "State", "observed_at": "Latest reading", "Risk band": st.column_config.TextColumn("Anomaly band"), "Anomaly score": st.column_config.NumberColumn("Score", format="%.4f")})
    st.caption("Model: scikit-learn Isolation Forest. Risk bands rank anomalous telemetry; they are not downtime probabilities.")


if __name__ == "__main__":
    main()
