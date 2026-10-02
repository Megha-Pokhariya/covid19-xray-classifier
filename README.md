# COVID-19 Chest X-Ray Classifier

A deep learning image classifier that categorizes chest X-ray images into three classes — **Covid**, **Normal**, and **Viral Pneumonia** — using transfer learning with MobileNetV2 (Keras/TensorFlow).

## 📌 Project Overview

This project trains a convolutional neural network (via transfer learning) to classify chest X-ray images. It includes:
- Exploratory data analysis (class distribution, sample image visualization)
- Data augmentation and preprocessing pipeline
- A MobileNetV2-based transfer learning model
- Model evaluation (accuracy, classification report, confusion matrix)
- A prediction function to classify new/unseen X-ray images

## ⚠️ Disclaimer

This is an **academic/educational machine learning project**, built as part of MBA coursework (AI & Data Science). It is **not a validated medical diagnostic tool** and should **not** be used for actual clinical diagnosis or treatment decisions. Predictions are for learning/demonstration purposes only.

## 📂 Dataset

- **Source:** [COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) by Pranav Raikokte, via Kaggle
- Used strictly for academic/educational purposes — not redistributed for any commercial use
- Original X-ray images were compiled from publicly available radiology sources (including Radiopaedia and similar medical imaging repositories)
- Classes: `Covid`, `Normal`, `Viral Pneumonia`
- Split: pre-divided into `train/` and `test/` folders

```
Covid19-dataset/
├── train/
│   ├── Covid/
│   ├── Normal/
│   └── Viral Pneumonia/
└── test/
    ├── Covid/
    ├── Normal/
    └── Viral Pneumonia/
```

> Full credit to the original dataset creator and the underlying public radiology image sources. If you are the rights holder of any image and have concerns about its inclusion here, please reach out and it will be addressed.

## 🛠️ Tech Stack

- Python, TensorFlow / Keras
- MobileNetV2 (pretrained on ImageNet) — transfer learning
- Matplotlib, Seaborn (visualization)
- Scikit-learn (evaluation metrics)
- Jupyter Lab / Notebook

## 🚀 How to Run

1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd covid19-xray-classifier
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the `Covid19-dataset/` folder (train/test) is in the project root
4. Open `covid19_xray_classifier.ipynb` in Jupyter Lab and run all cells top to bottom

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | _fill in after training_ |

(Add your confusion matrix / classification report screenshot here after training.)

## 🔮 Example Prediction

```python
predict_xray("Covid19-dataset/test/Covid/0100.jpeg")
# Output: Prediction: Covid (confidence: XX.X%)
```

## 👤 Author

**Megha Pokhariya**
MBA (AI & Data Science), Graphic Era University, Dehradun
[GitHub](https://github.com/Megha-Pokhariya)
