from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

# Store loaded model and metadata in application state
ml_state: Dict[str, Any] = {
    "model": None,
    "feature_names": [],
    "target_names": [],
    "loaded": False,
    "load_error": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading model artifact at startup."""
    try:
        if not MODEL_PATH.exists():
            ml_state["loaded"] = False
            ml_state["load_error"] = f"Model artifact not found at {MODEL_PATH}"
        else:
            artifact = joblib.load(MODEL_PATH)
            ml_state["model"] = artifact["model"]
            ml_state["feature_names"] = artifact["feature_names"]
            ml_state["target_names"] = artifact["target_names"]
            ml_state["loaded"] = True
            ml_state["load_error"] = None
    except Exception as exc:
        ml_state["loaded"] = False
        ml_state["load_error"] = str(exc)

    yield

    # Clean up on shutdown
    ml_state.clear()


app = FastAPI(
    title="Breast Cancer Wisconsin Classification API",
    description="Production-ready FastAPI service for predicting breast cancer malignancy from Wisconsin Diagnostic dataset features.",
    version="1.0.0",
    lifespan=lifespan,
)


class CancerFeatures(BaseModel):
    mean_radius: float = Field(..., alias="mean radius", description="Mean of distances from center to points on perimeter")
    mean_texture: float = Field(..., alias="mean texture", description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., alias="mean perimeter")
    mean_area: float = Field(..., alias="mean area")
    mean_smoothness: float = Field(..., alias="mean smoothness")
    mean_compactness: float = Field(..., alias="mean compactness")
    mean_concavity: float = Field(..., alias="mean concavity")
    mean_concave_points: float = Field(..., alias="mean concave points")
    mean_symmetry: float = Field(..., alias="mean symmetry")
    mean_fractal_dimension: float = Field(..., alias="mean fractal dimension")
    radius_error: float = Field(..., alias="radius error")
    texture_error: float = Field(..., alias="texture error")
    perimeter_error: float = Field(..., alias="perimeter error")
    area_error: float = Field(..., alias="area error")
    smoothness_error: float = Field(..., alias="smoothness error")
    compactness_error: float = Field(..., alias="compactness error")
    concavity_error: float = Field(..., alias="concavity error")
    concave_points_error: float = Field(..., alias="concave points error")
    symmetry_error: float = Field(..., alias="symmetry error")
    fractal_dimension_error: float = Field(..., alias="fractal dimension error")
    worst_radius: float = Field(..., alias="worst radius")
    worst_texture: float = Field(..., alias="worst texture")
    worst_perimeter: float = Field(..., alias="worst perimeter")
    worst_area: float = Field(..., alias="worst area")
    worst_smoothness: float = Field(..., alias="worst smoothness")
    worst_compactness: float = Field(..., alias="worst compactness")
    worst_concavity: float = Field(..., alias="worst concavity")
    worst_concave_points: float = Field(..., alias="worst concave points")
    worst_symmetry: float = Field(..., alias="worst symmetry")
    worst_fractal_dimension: float = Field(..., alias="worst fractal dimension")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )

    def to_feature_vector(self, feature_order: list[str]) -> list[float]:
        """Convert input model into exact feature vector order matching trained model schema."""
        feature_map = {
            "mean radius": self.mean_radius,
            "mean texture": self.mean_texture,
            "mean perimeter": self.mean_perimeter,
            "mean area": self.mean_area,
            "mean smoothness": self.mean_smoothness,
            "mean compactness": self.mean_compactness,
            "mean concavity": self.mean_concavity,
            "mean concave points": self.mean_concave_points,
            "mean symmetry": self.mean_symmetry,
            "mean fractal dimension": self.mean_fractal_dimension,
            "radius error": self.radius_error,
            "texture error": self.texture_error,
            "perimeter error": self.perimeter_error,
            "area error": self.area_error,
            "smoothness error": self.smoothness_error,
            "compactness error": self.compactness_error,
            "concavity error": self.concavity_error,
            "concave points error": self.concave_points_error,
            "symmetry error": self.symmetry_error,
            "fractal dimension error": self.fractal_dimension_error,
            "worst radius": self.worst_radius,
            "worst texture": self.worst_texture,
            "worst perimeter": self.worst_perimeter,
            "worst area": self.worst_area,
            "worst smoothness": self.worst_smoothness,
            "worst compactness": self.worst_compactness,
            "worst concavity": self.worst_concavity,
            "worst concave points": self.worst_concave_points,
            "worst symmetry": self.worst_symmetry,
            "worst fractal dimension": self.worst_fractal_dimension,
        }
        return [feature_map[name] for name in feature_order]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Numerical class prediction (0 or 1)")
    class_name: str = Field(..., alias="class", description="Human-readable diagnosis ('malignant' or 'benign')")

    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Returns the API health status and whether the ML model is currently loaded."""
    is_loaded = ml_state.get("loaded", False)
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Malignancy")
def predict(features: CancerFeatures):
    """Receives 30 numerical features and predicts breast cancer diagnosis."""
    if not ml_state.get("loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is unavailable: {ml_state.get('load_error', 'Model not loaded')}",
        )

    model = ml_state["model"]
    feature_order = ml_state["feature_names"]
    target_names = ml_state["target_names"]

    try:
        # Construct feature vector in deterministic order
        vector = [features.to_feature_vector(feature_order)]
        pred = int(model.predict(vector)[0])
        label = target_names[pred] if pred < len(target_names) else "unknown"

        return PredictionResponse(
            prediction=pred,
            class_name=label,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction execution failed: {str(exc)}",
        )
