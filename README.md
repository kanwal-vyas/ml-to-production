# ML-to-Production: Breast Cancer Classification

This repository contains an end-to-end Machine Learning model training and deployment pipeline for binary classification on the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## Project Structure

```text
ml-to-production/
├── main.py              # FastAPI application exposing /health, /predict, and /docs
├── train.py             # ML model training and export script
├── model.pkl            # Trained RandomForestClassifier artifact and feature metadata
├── requirements.txt     # Pinned Python package dependencies
├── test_api.py          # API test suite using FastAPI TestClient
├── README.md            # Project documentation and API guide
└── .gitignore           # Git ignore rules (tracks model.pkl)
```

---

## Stage 1: ML Model Training & Export

* **Dataset**: Breast Cancer Wisconsin (Diagnostic) (569 samples, 30 numerical features)
* **Model**: `RandomForestClassifier` (`random_state=42`)
* **Split**: 80% Train / 20% Test (Stratified, `random_state=42`)
* **Test Accuracy**: ~95.61%
* **Model Export**: `model.pkl` (saved with `joblib.dump()`, bundling model and feature names/classes)

Run training:
```bash
python train.py
```

---

## Stage 2: FastAPI Prediction Service

The API is built using **FastAPI** and **Pydantic v2**, loading `model.pkl` at startup and providing health checks and inference endpoints.

### 1. Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the API Locally

Start the development server with Uvicorn:

```bash
uvicorn main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

### 3. Interactive API Documentation

Once the server is running, explore and test the endpoints interactively via Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### 1. GET `/health`
Returns the status of the service and confirms whether the ML model artifact is loaded into memory.

#### Example Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 2. POST `/predict`
Accepts a JSON payload containing all 30 numerical features and returns the predicted class (`0` for `malignant`, `1` for `benign`).

#### Example Request:
```json
{
  "mean radius": 17.99,
  "mean texture": 10.38,
  "mean perimeter": 122.8,
  "mean area": 1001.0,
  "mean smoothness": 0.1184,
  "mean compactness": 0.2776,
  "mean concavity": 0.3001,
  "mean concave points": 0.1471,
  "mean symmetry": 0.2419,
  "mean fractal dimension": 0.07871,
  "radius error": 1.095,
  "texture error": 0.9053,
  "perimeter error": 8.589,
  "area error": 153.4,
  "smoothness error": 0.006399,
  "compactness error": 0.04904,
  "concavity error": 0.05373,
  "concave points error": 0.01587,
  "symmetry error": 0.03003,
  "fractal dimension error": 0.006193,
  "worst radius": 25.38,
  "worst texture": 17.33,
  "worst perimeter": 184.6,
  "worst area": 2019.0,
  "worst smoothness": 0.1622,
  "worst compactness": 0.6656,
  "worst concavity": 0.7119,
  "worst concave points": 0.2654,
  "worst symmetry": 0.4601,
  "worst fractal dimension": 0.1189
}
```

#### Example Response:
```json
{
  "prediction": 0,
  "class": "malignant"
}
```

#### Benign Example Request:
```json
{
  "mean radius": 13.54,
  "mean texture": 14.36,
  "mean perimeter": 87.46,
  "mean area": 566.3,
  "mean smoothness": 0.09779,
  "mean compactness": 0.08129,
  "mean concavity": 0.06664,
  "mean concave points": 0.04781,
  "mean symmetry": 0.1885,
  "mean fractal dimension": 0.05766,
  "radius error": 0.2699,
  "texture error": 0.7886,
  "perimeter error": 2.058,
  "area error": 23.56,
  "smoothness error": 0.008462,
  "compactness error": 0.0146,
  "concavity error": 0.02387,
  "concave points error": 0.01315,
  "symmetry error": 0.0198,
  "fractal dimension error": 0.0023,
  "worst radius": 15.11,
  "worst texture": 19.26,
  "worst perimeter": 99.7,
  "worst area": 711.2,
  "worst smoothness": 0.144,
  "worst compactness": 0.1773,
  "worst concavity": 0.239,
  "worst concave points": 0.1288,
  "worst symmetry": 0.2977,
  "worst fractal dimension": 0.07259
}
```

#### Benign Example Response:
```json
{
  "prediction": 1,
  "class": "benign"
}
```

---

## Running Tests

Run the test suite:

```bash
pytest
```
