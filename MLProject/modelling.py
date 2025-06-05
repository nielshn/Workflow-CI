# modelling.py
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(path):
    """Load dataset from given path."""
    print(f"[INFO] Loading dataset from: {path}")
    return pd.read_csv(path)


def train_model(X_train, y_train):
    """Train Random Forest model."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance."""
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='macro')
    rec = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')

    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] Precision: {prec:.4f}")
    print(f"[RESULT] Recall: {rec:.4f}")
    print(f"[RESULT] F1-score: {f1:.4f}")

    return acc, prec, rec, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default="winequality_preprocessed_train.csv",
        help="Path to preprocessed training data"
    )
    args = parser.parse_args()

    # Autolog all params, metrics, model
    mlflow.sklearn.autolog()

    # Load dataset
    df = load_data(args.data_path)
    X = df.drop("quality_label", axis=1)
    y = df["quality_label"]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run experiment
    with mlflow.start_run():  # Tidak pakai nested
        model = train_model(X_train, y_train)
        evaluate_model(model, X_val, y_val)
