# COVID-19 Chest X-Ray Classification System

An end-to-end deep learning project that classifies chest X-ray images into **COVID, Normal, and Viral Pneumonia** using MobileNetV2 transfer learning. The project combines model development and evaluation with a web-based prediction application built using FastAPI, HTML, CSS, and Docker.

## Project Overview

This project demonstrates an image classification workflow, from training and evaluating a deep learning model to deploying it through a local web application. Users can upload a chest X-ray image and view the predicted class, prediction confidence, and class-wise probabilities.

**Academic project | Deep Learning | Computer Vision | FastAPI | Docker**

## Key Features

- MobileNetV2 transfer learning with TensorFlow and Keras.
- Three-class chest X-ray classification.
- Test dataset evaluation using accuracy and classification metrics.
- Confusion matrix and class-wise performance analysis.
- FastAPI backend for model inference.
- Web interface with image upload and preview.
- Prediction confidence and class-wise probability display.
- Docker and Docker Compose support for local deployment.
- Interactive API documentation through Swagger UI.

## Model Performance

The trained model was evaluated on the notebook's test dataset.

| Evaluation Metric | Result |
|---|---:|
| Test images | 66 |
| Correct predictions | 62 |
| Incorrect predictions | 4 |
| **Test accuracy** | **93.94%** |
| Macro-average precision | 94.22% |
| Macro-average recall | 93.33% |
| Macro-average F1-score | 93.49% |
| Weighted-average F1-score | 93.91% |

### Class-Wise Classification Report

| Class | Precision | Recall | F1-score | Test Images |
|---|---:|---:|---:|---:|
| Covid | 96.30% | 100.00% | 98.11% | 26 |
| Normal | 100.00% | 85.00% | 91.89% | 20 |
| Viral Pneumonia | 86.36% | 95.00% | 90.48% | 20 |

**Note:** These metrics are from the saved outputs in the Jupyter Notebook. Performance may differ on new images and external datasets. The test set contains 66 images, so the reported accuracy should be interpreted in that context.

## Confusion Matrix

The confusion matrix compares the actual classes with the model's predicted classes.

| Actual Class | Predicted Covid | Predicted Normal | Predicted Viral Pneumonia |
|---|---:|---:|---:|
| Covid | 26 | 0 | 0 |
| Normal | 0 | 17 | 3 |
| Viral Pneumonia | 1 | 0 | 19 |

The model correctly classified all 26 Covid images in this test set. Three Normal images were classified as Viral Pneumonia, and one Viral Pneumonia image was classified as Covid.

## Technology Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Architecture:** MobileNetV2
- **Image Processing:** NumPy, Pillow
- **Model Evaluation:** Scikit-learn, Matplotlib, Seaborn
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Docker, Docker Compose
- **Development:** Jupyter Notebook, VS Code, GitHub

## Model and Image Preprocessing

The model uses MobileNetV2 pretrained on ImageNet as the transfer-learning backbone, with a custom classification head for three classes:

1. Covid
2. Normal
3. Viral Pneumonia

Input images are resized to **224 × 224 pixels**, converted to RGB, and normalized by dividing pixel values by 255. The trained Keras model is reused for inference when the web application starts; retraining is not required for each prediction.

## Project Architecture

```text
Chest X-Ray Image
        |
        v
HTML/CSS Web Interface
        |
        v
FastAPI Backend
        |
        v
Image Preprocessing
(224 x 224, RGB, /255.0)
        |
        v
Trained MobileNetV2 Model
        |
        v
Three-Class Prediction
        |
        v
Predicted Class + Confidence
+ Class-Wise Probabilities
```

Docker packages the application and its dependencies for local execution.

## Project Structure

```text
covid19-xray-classifier/
├── app/
│   ├── main.py
│   └── static/
│       └── index.html
├── models/
│   └── covid_pneumonia_classifier.keras
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── COVID19_X-RAY.ipynb
├── train.py
├── score.py
└── README.md
```

The notebook and training scripts support the machine learning workflow, while the `app` directory contains the web application.

## Run the Application Locally

### Prerequisites

- Git
- Docker Desktop with WSL integration enabled on Windows, or a compatible Docker Engine
- Access to the trained model file

### 1. Clone the Repository

```bash
git clone https://github.com/Megha-Pokhariya/covid19-xray-classifier.git
cd covid19-xray-classifier
```

### 2. Verify the Model File

Ensure that the following file exists:

```text
models/covid_pneumonia_classifier.keras
```

The model must be compatible with the class order and preprocessing expected by the application.

### 3. Build and Start the Application

```bash
docker compose up -d --build
```

### 4. Check the Container

```bash
docker compose ps
```

### 5. Open the Web Interface

Visit:

**http://localhost:8000**

Upload a chest X-ray image, preview it, and click the prediction button to view the result.

### 6. API Documentation

FastAPI's interactive documentation is available at:

**http://localhost:8000/docs**

### 7. Stop the Application

```bash
docker compose down
```

To start the application again later:

```bash
docker compose up -d
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Opens the web interface |
| GET | `/health` | Checks application health |
| POST | `/predict` | Accepts an image and returns prediction details |

## Understanding the Prediction

The application displays the predicted class, confidence score, and class-wise probabilities.

For this project's demonstration interface, the result mapping is:

- `Covid` → `COVID Positive`
- `Normal` → `COVID Negative`
- `Viral Pneumonia` → `COVID Negative`

These labels represent a simplified application rule, not a clinically validated diagnosis. A Normal or Viral Pneumonia prediction does not establish that a person is COVID-negative.

**Prediction confidence is not the same as test accuracy.** The reported 93.94% accuracy is calculated from the notebook's test evaluation, whereas an individual prediction's confidence applies only to that image.

## Dataset

The original machine learning project uses the COVID-19 Image Dataset from Kaggle, with the three classes used by the classifier. Refer to the original dataset source for licensing, terms of use, and attribution requirements.

## Limitations and Medical Disclaimer

This project is intended for academic and educational demonstration only. It is not a clinically validated medical diagnostic system. Its test dataset contains 66 images, and performance may not generalize to other populations, hospitals, imaging equipment, or datasets.

Predictions may be incorrect and must not be used to diagnose, exclude, or treat COVID-19, pneumonia, or any other medical condition. Consult qualified healthcare professionals for medical assessment.

## Author

**Megha Pokhariya**  
MBA — AI & Data Science  
Graphic Era University, Dehradun

GitHub: [Megha-Pokhariya](https://github.com/Megha-Pokhariya)
