import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from mlflow.models import infer_signature


class ModelTrainer:

    def __init__(self):
        self.feature_path = "data/features"
        self.model_path = "models"

    def train_model(self):

        # Set MLflow experiment
        mlflow.set_experiment("House Price Prediction")

        # Load datasets
        X_train = pd.read_csv(f"{self.feature_path}/X_train.csv")
        X_test = pd.read_csv(f"{self.feature_path}/X_test.csv")

        y_train = pd.read_csv(f"{self.feature_path}/y_train.csv").squeeze()
        y_test = pd.read_csv(f"{self.feature_path}/y_test.csv").squeeze()

        models = {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=90,
                random_state=42,
                n_jobs=-1
            )
        }

        best_model = None
        best_score = float("-inf")
        best_name = ""

        print("=" * 60)
        print("Training Models")
        print("=" * 60)

        for name, model in models.items():

            with mlflow.start_run(run_name=name):

                # Train model
                model.fit(X_train, y_train)

                # Prediction
                predictions = model.predict(X_test)

                # Evaluation
                score = r2_score(y_test, predictions)

                print(f"{name:<20} R² Score : {score:.4f}")

                # -----------------------------
                # Log Parameters
                # -----------------------------
                mlflow.log_param("Model", name)

                if name == "Random Forest":
                    mlflow.log_param("n_estimators", 90)
                    mlflow.log_param("random_state", 42)
                    mlflow.log_param("n_jobs", -1)

                # -----------------------------
                # Log Metrics
                # -----------------------------
                mlflow.log_metric("R2 Score", score)

                # -----------------------------
                # Model Signature
                # -----------------------------
                signature = infer_signature(
                    X_train,
                    model.predict(X_train)
                )

                # -----------------------------
                # Log Model
                # -----------------------------
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    signature=signature,
                    input_example=X_train.head(5)
                )

                # -----------------------------
                # Select Best Model
                # -----------------------------
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name

        # Save Best Model Locally
        os.makedirs(self.model_path, exist_ok=True)

        joblib.dump(
            best_model,
            f"{self.model_path}/best_model.pkl",compress=9
        )

        print("\n" + "=" * 60)
        print("Training Completed Successfully")
        print(f"Best Model     : {best_name}")
        print(f"Best R² Score  : {best_score:.4f}")
        print("Model Saved    : models/best_model.pkl")
        print("=" * 60)


if __name__ == "__main__":

    trainer = ModelTrainer()
    trainer.train_model()