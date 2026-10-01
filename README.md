# Smart Production Downtime Prediction App

A Streamlit dashboard for exploring underground mining equipment telemetry. The app loads the included CSV by default, or lets a user upload a telemetry CSV and train a fresh scikit-learn Isolation Forest for that dataset. The dashboard shows sensor trends, latest asset condition, a ranked watchlist, and model notes for whichever dataset is active.

## Run in VS Code

Open this folder in VS Code, open **Terminal → New Terminal**, and run:

```text
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens at the local URL shown in the terminal (usually `http://localhost:8501`). Keep the `data` folder beside `app.py` to use the built-in demo dataset.

## Train on your own telemetry

In the sidebar, upload a CSV. The app retrains the anomaly detector on the uploaded file and uses that dataset throughout the overview, asset detail, and model notes screens. It returns to the built-in demo when no file is uploaded.

Required columns (headers can be mapped in the app):

| Input | Expected unit |
| --- | --- |
| `asset_id` | Asset or machine identifier |
| `observed_at` | Timestamp parseable by pandas |
| `temperature_c` | °C |
| `vibration_mm_s` | mm/s |
| `pressure_kpa` | kPa |
| `current_a` | A |
| `rpm` | RPM |
| `uptime_percent` | Percent |

At least 20 valid rows are needed. Optional context columns are `asset_name`, `line`, and `operating_state`; missing context is filled automatically. Download the empty CSV template from the sidebar for the canonical header names.

## Dataset and model

The included `data/synthetic_underground_mining_telemetry.csv` contains 580 synthetic readings for five assets. The app trains an unsupervised Isolation Forest using temperature, vibration, pressure, current, RPM, uptime, and per-asset changes between readings. Risk bands rank readings relative to this dataset.

**This dataset does not contain downtime events:** every `operating_state` row is `RUNNING`, and the provided sensor values do not cross the listed warning thresholds. Consequently the trained model detects unusual telemetry patterns; it cannot learn downtime outcomes or estimate a validated downtime probability. The app labels this limitation in its interface. Use real historical sensor data joined to timestamped downtime records to train and validate a supervised downtime predictor.

Uploaded data uses the same unsupervised anomaly model. An `operating_state` or downtime label column is displayed as context but is not used as a target. Risk bands are relative to the dataset used to train the detector and are not probabilities of downtime.

The CSV was copied into this project from the file supplied for the task, so the project runs without relying on a path outside this folder.
