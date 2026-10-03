from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


def test_docs_endpoint():
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200


def test_predict_malignant_sample():
    malignant_payload = {
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
        "worst fractal dimension": 0.1189,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=malignant_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["class"] == "malignant"


def test_predict_benign_sample():
    benign_payload = {
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
        "worst fractal dimension": 0.07259,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=benign_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["class"] == "benign"


def test_predict_invalid_missing_field():
    incomplete_payload = {
        "mean radius": 13.54,
        "mean texture": 14.36,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=incomplete_payload)
        assert response.status_code == 422  # Unprocessable Entity
