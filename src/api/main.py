from fastapi import FastAPI, WebSocket
import json
import time
import asyncio
app = FastAPI()

FILE = "src/model_utils/inference_results.json"

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/detections/latest")
def latest():
    try:
        data = json.load(open(FILE))
        return data[-1] if data else {}
    except:
        return {}


@app.get("/detections/history")
def history():
    try:
        return json.load(open(FILE))
    except:
        return []


@app.websocket("/ws/detections")
async def ws(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            data = json.load(open(FILE))
        except:
            data = []

        await websocket.send_json(data)
        await asyncio.sleep(1)