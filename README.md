# COVID-19 Chest X-Ray Classifier

A deep learning image classification project that identifies **Covid-19**, **Viral Pneumonia**, or **Normal** condition from chest X-ray images, using transfer learning with MobileNetV2. Training was operationalized on **Azure Machine Learning** as a cloud-based ML job with a registered dataset and model registry.

## Overview

- **Task:** Multi-class image classification (3 classes) on chest X-ray images
- **Approach:** Transfer learning using a frozen MobileNetV2 backbone (pretrained on ImageNet) with a custom classification head
- **Test Accuracy:** 93.94%
- **Platform:** Trained via an Azure Machine Learning job, with the dataset registered as a versioned Data Asset and the trained model registered in the Azure ML Model Registry

## Dataset

Chest X-ray dataset split into `train` and `test` folders across three classes:

| Split | Covid | Normal | Viral Pneumonia |
|-------|-------|--------|------------------|
| Train | 111   | 70     | 70               |
| Test  | 26    | 20     | 20               |

## Model Architecture

```
MobileNetV2 (frozen, ImageNet weights)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, relu)
        ↓
Dropout(0.4)
        ↓
Dense(3, softmax)
```

- **Optimizer:** Adam (lr = 0.0001)
- **Loss:** Categorical Crossentropy
- **Class imbalance handling:** Class weights computed and applied during training (dataset had more Covid samples than Normal/Pneumonia)
- **Regularization:** Data augmentation (rotation, zoom, shift, horizontal flip) + Dropout
- **Epochs:** 15 (with EarlyStopping and ModelCheckpoint)

## Results

**Overall test accuracy: 93.94%**

| Class            | Precision | Recall | F1-score |
|-------------------|-----------|--------|----------|
| Covid              | 0.9630    | 1.0000 | 0.9811   |
| Normal             | 1.0000    | 0.8500 | 0.9189   |
| Viral Pneumonia    | 0.8636    | 0.9500 | 0.9048   |

Covid-19 cases were detected with **100% recall**, meaning no Covid case in the test set was missed by the model.

## Pipeline (Azure Machine Learning)

1. **Data Registration** — Dataset uploaded and registered as a versioned Azure ML Data Asset (`uri_folder` type)
2. **Training Script** — `train.py` — a standalone script that loads the registered dataset, builds the model, trains it, and saves the output
3. **Cloud Training Job** — Submitted as an Azure ML `command` job on a compute instance, using a TensorFlow-based curated environment
4. **Model Registration** — The trained model artifact was registered in the Azure ML Model Registry for versioning and traceability
5. **Inference** — Predictions are made via a reusable Python function (`predict_xray()`) that loads the saved model and classifies any input chest X-ray image with a confidence score

> **Note:** A real-time online endpoint deployment was also attempted using Azure ML Managed Online Endpoints. Due to compute and timeout constraints on the free/student subscription tier, the endpoint did not reliably serve predictions, so direct model inference (`predict_xray()`) was used instead as the primary prediction interface.

## Repository Structure

```
├── train.py                    # Training script (used in the Azure ML job)
├── score.py                    # Scoring script (used for online endpoint attempt)
├── COVID19_X-RAY.ipynb         # Full notebook: EDA, training, evaluation, inference
├── README.md
└── screenshots/                # Sample outputs (confusion matrix, predictions, Azure ML job)
```

> Dataset files and trained model weights (`.keras`) are not included in this repository due to size. See below for how to obtain/regenerate them.

## How to Run

1. Clone this repository
2. Place the chest X-ray dataset (train/test folders with Covid, Normal, Viral Pneumonia subfolders) in the project root as `Covid19-dataset/`
3. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn scikit-learn pillow
   ```
4. Run the notebook `COVID19_X-RAY.ipynb` to train the model and reproduce results
5. To predict on a new image:
   ```python
   predict_xray("path/to/xray_image.jpg")
   ```

## Tools & Technologies

Python, TensorFlow/Keras, MobileNetV2, scikit-learn, Azure Machine Learning (Data Assets, Jobs, Model Registry, Managed Online Endpoints)

## Author

**Megha Pokhariya**
MBA (AI & Data Science), Graphic Era University
[GitHub](https://github.com/Megha-Pokhariya)
