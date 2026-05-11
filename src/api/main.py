from fastapi import FastAPI, WebSocket
import json
import asyncio

# Create FastAPI application
app = FastAPI()

# Path to inference results file
FILE = "src/model_utils/inference_results.json"

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