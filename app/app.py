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

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from pathlib import Path
import base64

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import subsystems
    predict_acv = subsystems.predict_acv
    ACV_AVAILABLE = True
    ACV_IMPORT_ERROR = None
    SUBSYSTEMS_AVAILABLE = True
    SUBSYSTEMS_IMPORT_ERROR = None
except Exception as exc:
    subsystems = None
    predict_acv = None
    ACV_AVAILABLE = False
    ACV_IMPORT_ERROR = str(exc)
    SUBSYSTEMS_AVAILABLE = False
    SUBSYSTEMS_IMPORT_ERROR = str(exc)

SUBSYSTEM_PREDICT_FN = {}
SUBSYSTEM_FILE_EXT = {"acv": ".xlsx", "door": ".csv", "shm": ".csv", "corrugation": ".csv"}
if SUBSYSTEMS_AVAILABLE:
    SUBSYSTEM_PREDICT_FN = {
        "door": subsystems.predict_door,
        "shm": subsystems.predict_shm,
        "corrugation": subsystems.predict_rail,
    }

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
PORT = 8000

# Ensure proper MIME types
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("application/json", ".json")


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
            subsystems_status = {
                "acv": {
                    "available": ACV_AVAILABLE,
                    "status": "Ready" if ACV_AVAILABLE else f"Error: {ACV_IMPORT_ERROR}"
                }
            }
            for sub_id in ("door", "shm", "corrugation"):
                if SUBSYSTEMS_AVAILABLE:
                    try:
                        params = subsystems.get_model_params(sub_id)
                        subsystems_status[sub_id] = {
                            "available": True,
                            "status": "Ready",
                            "methodology": subsystems.METHODOLOGY[sub_id],
                            "params": params,
                        }
                    except Exception as exc:
                        subsystems_status[sub_id] = {"available": False, "status": f"Error: {exc}"}
                else:
                    subsystems_status[sub_id] = {"available": False, "status": f"Error: {SUBSYSTEMS_IMPORT_ERROR}"}

            self.send_json(200, {
                "status": "online",
                "system": "Train of Thought - Rail Condition Monitoring (CdM) Studio",
                "version": "2.1.0",
                "subsystems": subsystems_status
            })
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_json(400, {
                    "success": False,
                    "error": "Empty request body. Please upload a data file."
                })
                return

            raw_body = self.rfile.read(content_length)

            subsystem_id = self.headers.get("X-Subsystem")
            filename = self.headers.get("X-Filename", "acv_telemetry.xlsx")
            file_content = None

            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                    subsystem_id = payload.get("subsystem", subsystem_id)
                    filename = payload.get("filename", filename)
                    file_content = payload.get("file_content")
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

            if subsystem_id not in ("acv", "door", "shm", "corrugation"):
                self.send_json(400, {
                    "success": False,
                    "error": f"Unknown subsystem '{subsystem_id}'."
                })
                return

            if subsystem_id == "acv" and not ACV_AVAILABLE:
                self.send_json(500, {
                    "success": False,
                    "error": f"ACV model could not be imported: {ACV_IMPORT_ERROR}"
                })
                return
            if subsystem_id != "acv" and not SUBSYSTEMS_AVAILABLE:
                self.send_json(500, {
                    "success": False,
                    "error": f"Subsystem models could not be imported: {SUBSYSTEMS_IMPORT_ERROR}"
                })
                return

            # Decode file content (base64 or raw bytes)
            file_bytes = b""
            if isinstance(file_content, str):
                if file_content.startswith("data:") and ";base64," in file_content:
                    file_bytes = base64.b64decode(file_content.split(";base64,")[1])
                else:
                    try:
                        file_bytes = base64.b64decode(file_content)
                    except Exception:
                        file_bytes = file_content.encode("utf-8")
            elif isinstance(file_content, bytes):
                file_bytes = file_content

            if len(file_bytes) == 0:
                self.send_json(400, {
                    "success": False,
                    "error": "Received empty file content."
                })
                return

            suffix = os.path.splitext(filename)[1] or SUBSYSTEM_FILE_EXT.get(subsystem_id, ".csv")
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                if subsystem_id == "acv":
                    print(f"[CdM Server] Running predict_acv on '{filename}' ({tmp_path})...")
                    ranked_cars, scores = predict_acv(tmp_path)
                    ranking_string = "|".join(ranked_cars)
                    prediction_df = pd.DataFrame([{"file_id": filename, "ranked_cars": ranking_string}])

                    self.send_json(200, {
                        "success": True,
                        "subsystem": "acv",
                        "file_id": filename,
                        "faulty_car": ranked_cars[0],
                        "ranking": ranked_cars,
                        "ranking_string": ranking_string,
                        "scores": scores.to_dict(orient="records"),
                        "csv_data": prediction_df.to_csv(index=False)
                    })
                else:
                    print(f"[CdM Server] Running predict_{subsystem_id} on '{filename}' ({tmp_path})...")
                    result = SUBSYSTEM_PREDICT_FN[subsystem_id](tmp_path, display_name=filename)
                    self.send_json(200, {
                        "success": True,
                        "subsystem": subsystem_id,
                        "file_id": filename,
                        "methodology": subsystems.METHODOLOGY[subsystem_id],
                        "params": subsystems.get_model_params(subsystem_id),
                        **result,
                    })

            except Exception as exc:
                tb = traceback.format_exc()
                print(f"[CdM Server ERROR] {subsystem_id} prediction failure:", file=sys.stderr)
                print(tb, file=sys.stderr)

                self.send_json(422, {
                    "success": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": tb,
                    "subsystem": subsystem_id
                })

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
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
        print("Subsystems:      ACV | Door | Corrugation | SHM")
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
