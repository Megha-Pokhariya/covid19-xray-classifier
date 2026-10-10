from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model


app = FastAPI(title="COVID X-Ray Classifier")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "covid_pneumonia_classifier.keras"

CLASS_NAMES = ["Covid", "Normal", "Viral Pneumonia"]
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(
            __import__("io").BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize(IMG_SIZE)

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        probabilities = {
            CLASS_NAMES[i]: float(prediction[i] * 100)
            for i in range(len(CLASS_NAMES))
        }

        result = (
            "COVID Positive"
            if predicted_class == "Covid"
            else "COVID Negative"
        )

        return {
            "result": result,
            "predicted_class": predicted_class,
            "confidence_percent": round(confidence, 2),
            "class_probabilities": {
                k: round(v, 2) for k, v in probabilities.items()
            },
            "note": "Academic/demo project only. This is not a medical diagnosis."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
