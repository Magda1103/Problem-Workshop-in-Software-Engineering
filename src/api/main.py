from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import src.model_utils.inference_engine as inference_engine
import shutil
import json
import asyncio
import os
from pathlib import Path
import time

from src.model_utils.inference_engine import InferenceEngine
from src.model_utils.baseline_model import FRAME_STEP, FRAMES_COUNT

# Create FastAPI application
app = FastAPI()
UPLOAD_DIR = "data/uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CURRENT_ENGINE = None

# Path to inference results file
FILE = "src/model_utils/inference_results.json"
EVENTS_FILE = "src/model_utils/alert_events.json"

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/detections/latest")
def latest():
    """
    Returns latest detection result from json file
    """
    try:
        data = json.load(open(FILE))
        return data[-1] if data else {}
    except:
        return {}


@app.get("/detections/history")
def history():
    """
    Returns all saved detection history from json file
    """
    try:
        return json.load(open(FILE))
    except:
        return []


@app.get("/alert-events")
def alert_events():
    """
    Returns alert state change history
    """
    try:
        return json.load(open(EVENTS_FILE))
    except:
        return []


@app.websocket("/ws/detections")
async def ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detections
    sends updated detection data every second
    """
    await websocket.accept()

    while True:
        try:
            data = json.load(open(FILE))
        except:
            data = []

        await websocket.send_json(data)
        await asyncio.sleep(1)


def run_ai_analysis(video_path_str: str):
    """
    Background task to initialize and run the InferenceEngine for the uploaded file.
    """
    global CURRENT_ENGINE

    with open(FILE, "w") as f:
        json.dump([], f)

    if CURRENT_ENGINE is not None:
        print("Stopping previous AI analysis...")
        CURRENT_ENGINE.stop_event.set()
        time.sleep(1)

    inference_engine.LATEST_FRAME = None  # Clear the previous frame buffer
    print(f"Starting AI analysis for file: {video_path_str}")
    video_path = Path(video_path_str)

    # Initialize the engine with configured settings
    CURRENT_ENGINE = InferenceEngine(frame_step=FRAME_STEP, frames_limit=FRAMES_COUNT, video_path=video_path)
    CURRENT_ENGINE.perform_inference()

    print(f"Finished AI analysis for file: {video_path_str}")
    CURRENT_ENGINE = None


async def frame_generator():
    """Generator yielding the latest rendered frame with bounding boxes."""
    while True:
        if inference_engine.LATEST_FRAME is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + inference_engine.LATEST_FRAME + b'\r\n')
        await asyncio.sleep(0.05)  # Target roughly 20 frames per second


@app.get("/video_feed")
async def video_feed():
    """Endpoint serving the live video stream (MJPEG)."""
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# File upload endpoint
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Dispatch the background task
    background_tasks.add_task(run_ai_analysis, file_path)

    return {
        "filename": file.filename,
        "message": "File uploaded successfully. AI analysis started in the background!"
    }


app.mount("/videos", StaticFiles(directory=UPLOAD_DIR), name="videos")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")