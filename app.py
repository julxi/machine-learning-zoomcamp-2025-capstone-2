import onnxruntime as ort
import numpy as np
import math
from preprocess import board_to_array
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
session = ort.InferenceSession(
    "models/chess_eval-full_data-medium_model-2_epochs.onnx",
    providers=["CPUExecutionProvider"],
)


def predict_fen(fen: str) -> float:
    x = board_to_array(fen).astype(np.float32)
    x = np.expand_dims(x, axis=0)  # batch dim

    y_transformed = session.run(
        None,
        {"board": x},
    )[
        0
    ][0]

    # inverse transform
    return 100.0 * math.sinh(float(y_transformed))


app = FastAPI()


class FenRequest(BaseModel):
    fen: str


@app.post("/predict")
def predict(req: FenRequest):
    value = predict_fen(req.fen)
    return {"evaluation": value}


@app.get("/health")
def health_check():
    try:
        # Optionally, you can add a lightweight model inference here to check model health
        dummy_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        predict_fen(dummy_fen)
        return {"status": "healthy", "model": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
