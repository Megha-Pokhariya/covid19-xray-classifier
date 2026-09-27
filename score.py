import json
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import base64
from io import BytesIO

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "covid_pneumonia_classifier.keras")
    model = load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)
    image_data = base64.b64decode(data["image"])
    img = Image.open(BytesIO(image_data)).convert("RGB").resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    class_names = ["Covid", "Normal", "Viral Pneumonia"]
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction[0]) * 100)

    return json.dumps({
        "predicted_class": predicted_class,
        "confidence": confidence
    })
