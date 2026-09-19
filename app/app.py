#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
import json
import mimetypes
import tempfile
import traceback
import math
import io
import base64
import zipfile
from pathlib import Path

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
REPO_ROOT = Path(PROJECT_ROOT)
PORT = 8000

# Ensure proper MIME types
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("application/json", ".json")

# -----------------------------------------------------------------------------
# Optional Imports: Pandas & Numpy
# -----------------------------------------------------------------------------
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except Exception as exc:
    pd = None
    np = None
    PANDAS_AVAILABLE = False
    print(f"[Warning] Pandas/Numpy not available: {exc}", file=sys.stderr)

# -----------------------------------------------------------------------------
# ACV Model Integration (Optional_Items/ACV/code)
# -----------------------------------------------------------------------------
ACV_CODE_ROOT = REPO_ROOT / "Optional_Items" / "ACV" / "code"
if str(ACV_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACV_CODE_ROOT))

try:
    from src.pipeline import predict_acv
    ACV_AVAILABLE = True
    ACV_IMPORT_ERROR = None
except Exception as exc:
    predict_acv = None
    ACV_AVAILABLE = False
    ACV_IMPORT_ERROR = str(exc)

# -----------------------------------------------------------------------------
# Helper: Decode Base64 / Bytes
# -----------------------------------------------------------------------------
def decode_file_bytes(file_content):
    if isinstance(file_content, bytes):
        return file_content
    if isinstance(file_content, str):
        if file_content.startswith("data:") and ";base64," in file_content:
            return base64.b64decode(file_content.split(";base64,")[1])
        try:
            return base64.b64decode(file_content)
        except Exception:
            return file_content.encode("utf-8")
    return b""

# -----------------------------------------------------------------------------
# Door Subsystem Engine (models/Door/door_model.json)
# -----------------------------------------------------------------------------
DOOR_MODEL_PATH = REPO_ROOT / "models" / "Door" / "door_model.json"

def predict_door(file_bytes, filename="continuous_stream.csv"):
    if not DOOR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Door model definition not found at {DOOR_MODEL_PATH}")

    with open(DOOR_MODEL_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    df = pd.read_csv(io.BytesIO(file_bytes))
    required_cols = ["Datetime", "Motor current(mA)", "Close command", "Open command"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in Door telemetry CSV. Found columns: {list(df.columns)}")

    parts = df["Datetime"].astype(str).str.split("-", expand=True).astype(int)
    parts.columns = ["Y", "Mo", "D", "H", "Mi", "S", "Ms"]
    ts = (((parts["D"] * 24 + parts["H"]) * 60 + parts["Mi"]) * 60 + parts["S"]) * 1000 + parts["Ms"]
    deltas = ts.diff()
    gap_threshold = model_info.get("gap_threshold_ms", 1000)
    boundaries = [0] + list(deltas[deltas > gap_threshold].index) + [len(df)]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    predictions = []
    normal_count = 0
    abnormal_count = 0

    for s, e in segments:
        chunk = df.iloc[s:e]
        if len(chunk) == 0:
            continue
        op = "Close" if (chunk["Close command"] == 1).mean() > (chunk["Open command"] == 1).mean() else "Open"
        current_mean = float(chunk["Motor current(mA)"].mean())
        op_model = model_info["operations"][op]
        z = op_model["coef"] * current_mean + op_model["intercept"]
        prob_abnormal = 1.0 / (1.0 + math.exp(-z))
        prediction = "Abnormal resistance" if prob_abnormal >= 0.5 else "Normal"

        if prediction == "Abnormal resistance":
            abnormal_count += 1
        else:
            normal_count += 1

        predictions.append({
            "start_time": str(chunk.iloc[0]["Datetime"]),
            "end_time": str(chunk.iloc[-1]["Datetime"]),
            "prediction": prediction,
            "operation": op,
            "current_mean": round(current_mean, 2)
        })

    out_df = pd.DataFrame(predictions)[["start_time", "end_time", "prediction"]]
    csv_data = out_df.to_csv(index=False)

    status = "warning" if abnormal_count > 0 else "healthy"
    headline = f"{abnormal_count} Abnormal Resistance Cycles Detected" if abnormal_count > 0 else "All Door Cycles Normal"
    summary = (
        f"Segmented {len(predictions)} operation cycles ({normal_count} Normal, {abnormal_count} Abnormal resistance). "
        f"Model: Logistic Regression boundary on mean motor current (gap threshold = {gap_threshold}ms)."
    )

    return {
        "success": True,
        "subsystem": "door",
        "file_id": filename,
        "status": status,
        "headline": headline,
        "summary": summary,
        "total_cycles": len(predictions),
        "abnormal_cycles": abnormal_count,
        "normal_cycles": normal_count,
        "predictions": predictions,
        "csv_data": csv_data
    }

# -----------------------------------------------------------------------------
# SHM Subsystem Engine (models/SHM/shm_model.json)
# -----------------------------------------------------------------------------
SHM_MODEL_PATH = REPO_ROOT / "models" / "SHM" / "shm_model.json"

def turning_points(x):
    x = np.asarray(x, dtype=np.float64)
    diffs = np.diff(x)
    nonzero_idx = np.flatnonzero(diffs != 0)
    if len(nonzero_idx) < 2:
        return x[[0, -1]] if len(x) >= 2 else x
    d = diffs[nonzero_idx]
    sign_change = np.where(d[:-1] * d[1:] < 0)[0] + 1
    keep = np.concatenate(([nonzero_idx[0]], nonzero_idx[sign_change], [nonzero_idx[-1] + 1]))
    return x[keep]

def count_cycles(x):
    points = turning_points(x)
    stack = []
    cycles = []
    for p in points:
        stack.append(p)
        while len(stack) >= 4:
            x1, x2, x3, x4 = stack[-4], stack[-3], stack[-2], stack[-1]
            y1, y2, y3 = abs(x1 - x2), abs(x2 - x3), abs(x3 - x4)
            if y2 <= y1 and y2 <= y3:
                cycles.append((y2, (x2 + x3) / 2.0, 1.0))
                del stack[-3:-1]
            else:
                break
    for j in range(len(stack) - 1):
        rng = abs(stack[j] - stack[j + 1])
        mean = (stack[j] + stack[j + 1]) / 2.0
        cycles.append((rng, mean, 0.5))
    return cycles

def predict_single_shm_series(series, model_info):
    if model_info.get("type") == "power_law":
        m = model_info["m"]
        a = model_info["a"]
        cycles = count_cycles(series)
        ranges = np.array([c[0] for c in cycles])
        counts = np.array([c[2] for c in cycles])
        amps = ranges / 2.0
        F_m = float(np.sum(counts * (amps ** m))) if len(amps) else 0.0
        return float(a * F_m)
    raise ValueError(f"Unsupported SHM model type: {model_info.get('type')}")

def predict_shm(file_bytes, filename="stress_series.csv", file_list=None):
    if not SHM_MODEL_PATH.exists():
        raise FileNotFoundError(f"SHM model definition not found at {SHM_MODEL_PATH}")

    with open(SHM_MODEL_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    predictions = []

    # Case A: Multiple files sent via JSON list
    if file_list and isinstance(file_list, list) and len(file_list) > 0:
        for item in file_list:
            fname = item.get("filename", "unknown.csv")
            fbytes = decode_file_bytes(item.get("file_content", ""))
            if len(fbytes) == 0:
                continue
            series = pd.read_csv(io.BytesIO(fbytes), header=None).iloc[:, 0].to_numpy()
            pred = predict_single_shm_series(series, model_info)
            predictions.append({"file_id": fname, "prediction": round(pred, 4)})

    # Case B: Zip file containing multiple test CSV files
    elif filename.lower().endswith(".zip") or (isinstance(file_bytes, bytes) and file_bytes[:4] == b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            csv_names.sort()
            for cname in csv_names:
                with z.open(cname) as zf:
                    series = pd.read_csv(zf, header=None).iloc[:, 0].to_numpy()
                    pred = predict_single_shm_series(series, model_info)
                    predictions.append({"file_id": os.path.basename(cname), "prediction": round(pred, 4)})

    # Case C: Single CSV file
    else:
        series = pd.read_csv(io.BytesIO(file_bytes), header=None).iloc[:, 0].to_numpy()
        pred = predict_single_shm_series(series, model_info)
        predictions.append({"file_id": filename, "prediction": round(pred, 4)})

    out_df = pd.DataFrame(predictions)[["file_id", "prediction"]]
    csv_data = out_df.to_csv(index=False)

    max_damage = max(p["prediction"] for p in predictions) if predictions else 0.0
    mean_damage = sum(p["prediction"] for p in predictions) / len(predictions) if predictions else 0.0

    status = "critical" if max_damage >= 0.8 else ("warning" if max_damage >= 0.3 else "healthy")
    headline = f"Peak Fatigue Damage: {max_damage:.4f}"
    summary = (
        f"Evaluated {len(predictions)} stress series via ASTM E1049 Rainflow cycle counting and Miner's rule "
        f"(S-N power law exponent m={model_info.get('m')}, a={model_info.get('a'):.3e}, LOO score={model_info.get('loo_score'):.4f})."
    )

    return {
        "success": True,
        "subsystem": "shm",
        "file_id": filename,
        "status": status,
        "headline": headline,
        "summary": summary,
        "total_files": len(predictions),
        "max_damage": round(max_damage, 4),
        "mean_damage": round(mean_damage, 4),
        "predictions": predictions,
        "csv_data": csv_data
    }

# -----------------------------------------------------------------------------
# HTTP Server Handler
# -----------------------------------------------------------------------------
class CdMServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=APP_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Subsystem, X-Filename")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, status_code: int, data: dict):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            self.send_json(200, {
                "status": "online",
                "system": "Train of Thought - Rail Condition Monitoring (CdM) Studio",
                "version": "2.1.0",
                "subsystems": {
                    "door": {
                        "available": PANDAS_AVAILABLE and DOOR_MODEL_PATH.exists(),
                        "status": "Ready (Logistic Regression, gap=1000ms)" if DOOR_MODEL_PATH.exists() else "Model file missing"
                    },
                    "acv": {
                        "available": ACV_AVAILABLE,
                        "status": "Ready (Fault Localisation)" if ACV_AVAILABLE else f"Error: {ACV_IMPORT_ERROR}"
                    },
                    "shm": {
                        "available": PANDAS_AVAILABLE and SHM_MODEL_PATH.exists(),
                        "status": "Ready (Power Law, m=5, a=1.368e-9)" if SHM_MODEL_PATH.exists() else "Model file missing"
                    },
                    "corrugation": {
                        "available": False,
                        "status": "Model not implemented"
                    }
                }
            })
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_json(400, {
                    "success": False,
                    "error": "Empty request body. Please upload a telemetry file."
                })
                return

            raw_body = self.rfile.read(content_length)

            subsystem_id = self.headers.get("X-Subsystem")
            filename = self.headers.get("X-Filename", "telemetry_input.csv")
            file_content = None
            file_list = None

            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                    subsystem_id = payload.get("subsystem", subsystem_id)
                    filename = payload.get("filename", filename)
                    file_content = payload.get("file_content")
                    file_list = payload.get("files")
                except Exception as e:
                    self.send_json(400, {"success": False, "error": f"Invalid JSON payload: {str(e)}"})
                    return
            else:
                file_content = raw_body

            if not subsystem_id:
                self.send_json(400, {
                    "success": False,
                    "error": "Missing subsystem ID. Please specify 'subsystem'."
                })
                return

            subsystem_id = subsystem_id.strip().lower()

            # Reject Corrugation as not implemented
            if subsystem_id == "corrugation":
                self.send_json(400, {
                    "success": False,
                    "error": "Model not implemented for Rail Corrugation subsystem."
                })
                return

            if not PANDAS_AVAILABLE:
                self.send_json(500, {
                    "success": False,
                    "error": "Pandas/Numpy are required to run models on the server."
                })
                return

            # Dispatch to appropriate model engine
            try:
                # -------------------------------------------------------------
                # 1. ACV Subsystem
                # -------------------------------------------------------------
                if subsystem_id == "acv":
                    if not ACV_AVAILABLE:
                        raise RuntimeError(f"ACV model could not be imported: {ACV_IMPORT_ERROR}")

                    file_bytes = decode_file_bytes(file_content)
                    if len(file_bytes) == 0:
                        raise ValueError("Received empty file content for ACV.")

                    suffix = os.path.splitext(filename)[1] or ".xlsx"
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        print(f"[CdM Server] Running predict_acv on '{filename}'...")
                        ranked_cars, scores = predict_acv(tmp_path)

                        faulty_car = ranked_cars[0]
                        ranking_string = "|".join(ranked_cars)

                        prediction_df = pd.DataFrame([{
                            "file_id": filename,
                            "ranked_cars": ranking_string
                        }])
                        csv_data = prediction_df.to_csv(index=False)

                        self.send_json(200, {
                            "success": True,
                            "subsystem": "acv",
                            "file_id": filename,
                            "status": "critical",
                            "headline": f"Suspected Faulty Car: Car {faulty_car}",
                            "summary": f"Full Ranking: {ranking_string}",
                            "faulty_car": faulty_car,
                            "ranking": ranked_cars,
                            "ranking_string": ranking_string,
                            "scores": scores.to_dict(orient="records"),
                            "csv_data": csv_data
                        })
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                    return

                # -------------------------------------------------------------
                # 2. Door Subsystem
                # -------------------------------------------------------------
                elif subsystem_id == "door":
                    file_bytes = decode_file_bytes(file_content)
                    if len(file_bytes) == 0:
                        raise ValueError("Received empty file content for Door.")

                    print(f"[CdM Server] Running Door segmentation and classification on '{filename}'...")
                    result = predict_door(file_bytes, filename=filename)
                    self.send_json(200, result)
                    return

                # -------------------------------------------------------------
                # 3. SHM Subsystem
                # -------------------------------------------------------------
                elif subsystem_id == "shm":
                    file_bytes = decode_file_bytes(file_content)
                    if len(file_bytes) == 0 and not file_list:
                        raise ValueError("Received empty file content for SHM.")

                    print(f"[CdM Server] Running SHM rainflow and fatigue estimation on '{filename}'...")
                    result = predict_shm(file_bytes, filename=filename, file_list=file_list)
                    self.send_json(200, result)
                    return

                else:
                    self.send_json(400, {
                        "success": False,
                        "error": f"Unknown subsystem '{subsystem_id}'."
                    })
                    return

            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[CdM Server ERROR] Prediction failure on '{subsystem_id}':", file=sys.stderr)
                print(tb, file=sys.stderr)

                self.send_json(422, {
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": tb,
                    "subsystem": subsystem_id
                })
            return

        super().do_POST()


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CdMServerHandler) as httpd:
        print("=" * 72)
        print("🚆  TRAIN OF THOUGHT - RAIL CONDITION MONITORING (CdM) STUDIO")
        print("=" * 72)
        print(f"Web Interface:   http://localhost:{PORT}")
        print(f"Backend API:     http://localhost:{PORT}/api/predict")
        print(f"Model Status:    http://localhost:{PORT}/api/status")
        print("Subsystems:      Door (Active) | ACV (Active) | SHM (Active)")
        print("                 Rail Corrugation (Not Implemented)")
        print("-" * 72)
        print("Press Ctrl+C to stop the server.")
        print("=" * 72)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down CdM server...")
            httpd.server_close()

if __name__ == "__main__":
    run_server()
