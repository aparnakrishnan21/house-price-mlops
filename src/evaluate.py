import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluation:
    def __init__(self):
        self.model_path = "models/best_model.pkl"
        self.feature_path = "data/features"

    def evaluate_model(self):
        # Load model
        model = joblib.load(self.model_path)

        # Load test data
        X_test = pd.read_csv(f"{self.feature_path}/X_test.csv")
        y_test = pd.read_csv(f"{self.feature_path}/y_test.csv").squeeze()

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        print("=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        print(f"Mean Absolute Error : {mae:,.2f}")
        print(f"Root Mean Squared Error : {rmse:,.2f}")
        print(f"R² Score : {r2:.4f}")
        print("=" * 60)

        # Save metrics
        os.makedirs("reports", exist_ok=True)

        metrics = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2_Score": float(r2)
        }

        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # -----------------------
        # Plot 1: Actual vs Predicted
        # -----------------------
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted House Prices")
        plt.savefig("reports/actual_vs_predicted.png")
        plt.close()

         # -----------------------------
        # Residual Plot
        # -----------------------------
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.savefig("reports/residual_plot.png")
        plt.show()

        # -----------------------------
        # Residual Distribution Plot
        # -----------------------------    

        try:

            plt.figure(figsize=(8, 6))

            sns.histplot(
                residuals,
                bins=30,
                kde=True
            )

            plt.xlabel("Residual")
            plt.title("Residual Distribution")
            plt.tight_layout()
            plt.savefig("reports/residual_distribution.png")
            plt.close()

            print("✓ residual_distribution.png saved")

        except Exception as e:
            print("Residual Distribution Error:", e)

        # -----------------------------
        # Feature Importance Plot
        # -----------------------------     

        if hasattr(model, "feature_importances_"):

           try:

               importance = pd.DataFrame({
               "Feature": X_test.columns,
               "Importance": model.feature_importances_
               })

               importance = importance.sort_values(
               by="Importance",
               ascending=False
               )

               plt.figure(figsize=(10, 6))

               sns.barplot(
                data=importance,
                x="Importance",
                y="Feature"
                )

               plt.title("Feature Importance")

               plt.tight_layout()

               plt.savefig("reports/feature_importance.png")
               plt.close()

               print("✓ feature_importance.png saved")

           except Exception as e:
            print("Feature Importance Error:", e)

        else:
            print("Current model does not support feature_importances_")

        print("\nEvaluation Completed Successfully")

        print("Metrics saved to reports/metrics.json")


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate_model()  