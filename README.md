# ML-to-Production: Breast Cancer Classification

This repository contains an end-to-end Machine Learning model training and deployment pipeline.

## Stage 1: ML Model Training & Export

In Stage 1, a binary classification model is trained on scikit-learn's built-in Breast Cancer Wisconsin dataset using a `RandomForestClassifier`.

### Model Pipeline Summary
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) dataset (569 samples, 30 features)
- **Algorithm**: `RandomForestClassifier` (`random_state=42`)
- **Data Split**: 80% Train / 20% Test (Stratified, `random_state=42`)
- **Artifact Export**: `model.pkl` (contains trained model, feature ordering, and class names via `joblib`)

### Running Training Script

```bash
python train.py
```
