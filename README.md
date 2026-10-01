# Smart Production Downtime Prediction App

A Streamlit dashboard for exploring underground mining equipment telemetry. The app loads the included CSV and trains a scikit-learn Isolation Forest when it starts. The model detects sensor patterns that are unusual compared with this dataset and shows the latest condition for each asset.

## Run in VS Code

Open this folder in VS Code, open **Terminal → New Terminal**, and run:

```text
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens at the local URL shown in the terminal (usually `http://localhost:8501`). Keep the `data` folder beside `app.py` so the app can find the training CSV.

## Dataset and model

The included `data/synthetic_underground_mining_telemetry.csv` contains 580 synthetic readings for five assets. The app trains an unsupervised Isolation Forest using temperature, vibration, pressure, current, RPM, uptime, and per-asset changes between readings. Risk bands rank readings relative to this dataset.

**This dataset does not contain downtime events:** every `operating_state` row is `RUNNING`, and the provided sensor values do not cross the listed warning thresholds. Consequently the trained model detects unusual telemetry patterns; it cannot learn downtime outcomes or estimate a validated downtime probability. The app labels this limitation in its interface. Use real historical sensor data joined to timestamped downtime records to train and validate a supervised downtime predictor.

The CSV was copied into this project from the file supplied for the task, so the project runs without relying on a path outside this folder.
