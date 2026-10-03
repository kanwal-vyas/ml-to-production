import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def main():
    # 1. Load dataset
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()

    # 2. Inspect dataset properties
    n_samples, n_features = data.data.shape
    target_classes = [str(c) for c in data.target_names]
    feature_names = [str(f) for f in data.feature_names]

    print("\n--- Dataset Overview ---")
    print(f"Number of samples: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Target classes: {target_classes}")
    print(f"Feature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names, start=1):
        print(f"  {i:2d}. {name}")

    # 3. Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    print("\n--- Data Split ---")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    # 4. Train RandomForestClassifier
    print("\nTraining RandomForestClassifier (random_state=42)...")
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate on Test Set
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 6. Print Evaluation Results
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 7. Export trained model artifact and feature metadata
    model_filepath = "model.pkl"
    artifact = {
        "model": clf,
        "feature_names": feature_names,
        "target_names": target_classes,
    }

    joblib.dump(artifact, model_filepath)
    print(f"\nTrained model and metadata exported successfully to '{model_filepath}'.")

    # 8. Verification: Reload model artifact
    loaded_artifact = joblib.load(model_filepath)
    loaded_model = loaded_artifact["model"]
    loaded_features = loaded_artifact["feature_names"]

    # Verify prediction with loaded model
    loaded_pred = loaded_model.predict(X_test)
    assert (loaded_pred == y_pred).all(), "Loaded model predictions do not match trained model!"
    assert loaded_features == feature_names, "Saved feature names do not match!"
    print("Verification passed: Saved model loaded cleanly and verified with joblib.load().")


if __name__ == "__main__":
    main()
