# COVID-19 Chest X-Ray Classification System

An end-to-end deep learning project that classifies chest X-ray images into **Covid, Normal, and Viral Pneumonia** using MobileNetV2 transfer learning. The project combines machine learning model development with a web-based prediction application built using FastAPI, HTML, CSS, and Docker.

## Project Overview

This project was developed as part of academic work in AI and Data Science. It demonstrates the workflow from image classification model development to local web application deployment.

### Key Features

- MobileNetV2-based image classification using TensorFlow and Keras.
- Three-class prediction: Covid, Normal, and Viral Pneumonia.
- Model development and evaluation using a Jupyter Notebook.
- FastAPI backend for image prediction.
- Web interface for uploading chest X-ray images.
- Image preview before prediction.
- Prediction confidence and class-wise probability display.
- Docker-based application deployment.
- Local browser access and optional access from a phone on the same Wi-Fi network.

## Project Architecture

```text
Chest X-Ray Image
        |
        v
Web Interface (HTML/CSS)
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
+ Class Probabilities
        |
        v
Application Result Label
```

Docker packages the application and its dependencies for local execution.

## Technology Stack

- Python
- TensorFlow and Keras
- MobileNetV2
- FastAPI and Uvicorn
- HTML and CSS
- NumPy and Pillow
- Docker and Docker Compose
- Azure Machine Learning
- Jupyter Notebook

## Machine Learning Model

The classifier uses transfer learning with MobileNetV2 pretrained on ImageNet. A custom classification head predicts one of three classes:

1. `Covid`
2. `Normal`
3. `Viral Pneumonia`

Input images are resized to 224 × 224 pixels, converted to RGB, and normalized by dividing pixel values by 255.

The application reuses the trained Keras model for inference. Retraining is not required each time the application starts.

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

The structure above describes the intended application layout. The notebook and training scripts are retained as part of the original machine learning workflow.

## Run the Application Locally

### Prerequisites

- Docker Desktop with WSL integration enabled, or a compatible Docker Engine.
- The trained Keras model file.
- Git, if cloning the repository.

### 1. Clone the Repository

```bash
git clone https://github.com/Megha-Pokhariya/covid19-xray-classifier.git
cd covid19-xray-classifier
```

### 2. Check the Model File

Ensure the trained model is present at:

```text
models/covid_pneumonia_classifier.keras
```

The model must match the class names and image preprocessing expected by the application.

### 3. Start the Docker Application

```bash
docker compose up -d --build
```

### 4. Check the Running Container

```bash
docker compose ps
```

### 5. Open the Web Application

Open the following URL in your laptop browser:

**http://localhost:8000**

Upload a chest X-ray image, preview it, and click the prediction button to view the result.

### 6. Access from a Phone

Connect your phone and laptop to the same Wi-Fi network. Find your laptop's local IPv4 address and open the following URL on your phone:

```text
http://YOUR-LAPTOP-IP:8000
```

Replace `YOUR-LAPTOP-IP` with your laptop's actual local IPv4 address. Firewall settings may need to allow access to port 8000.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Opens the web interface |
| GET | `/health` | Checks application health |
| POST | `/predict` | Accepts an image and returns prediction details |

Interactive API documentation is available at:

**http://localhost:8000/docs**

## Understanding the Prediction

The application displays the predicted class, confidence score, and class-wise probabilities.

The application's simplified result mapping is:

- `Covid` → **COVID Positive**
- `Normal` → **COVID Negative**
- `Viral Pneumonia` → **COVID Negative**

These labels follow a project-specific demonstration rule. A `Viral Pneumonia` or `Normal` prediction does not establish that a person is medically COVID-negative.

## Stop the Application

To stop the Docker application:

```bash
docker compose down
```

To start it again later:

```bash
docker compose up -d
```

## Dataset

The original machine learning project uses the [COVID-19 Image Dataset on Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

The dataset contains the three classes used by the classifier. Refer to the original dataset source for its terms of use and attribution requirements.

## Academic Disclaimer

This project is intended **for academic and educational demonstration only**. It is not a clinically validated medical diagnostic system. Predictions may be inaccurate and must not be used to diagnose, exclude, or treat COVID-19 or any other medical condition. Consult qualified healthcare professionals for medical assessment.

## Author

**Megha Pokhariya**  
MBA — AI & Data Science  
Graphic Era University, Dehradun

GitHub: [Megha-Pokhariya](https://github.com/Megha-Pokhariya)
