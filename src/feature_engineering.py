# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# class FeatureEngineering:
#     def __init__(self):
#         self.input_path = "data/processed/processed_data.csv"
#         self.output_dir = "data/features"

#     def engineer_features(self):
#         # Load processed dataset
#         df = pd.read_csv(self.input_path)

#         # Separate features and target
#         X = df.drop("price", axis=1)
#         y = df["price"]

#         # Train-Test Split
#         X_train, X_test, y_train, y_test = train_test_split(
#             X,
#             y,
#             test_size=0.2,
#             random_state=42
#         )

#         # Scale numeric columns
#         scaler = StandardScaler()

#         numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns

#         X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
#         X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

#         # Create output directory
#         os.makedirs(self.output_dir, exist_ok=True)

#         # Save datasets
#         X_train.to_csv(f"{self.output_dir}/X_train.csv", index=False)
#         X_test.to_csv(f"{self.output_dir}/X_test.csv", index=False)

#         y_train.to_csv(f"{self.output_dir}/y_train.csv", index=False)
#         y_test.to_csv(f"{self.output_dir}/y_test.csv", index=False)

#         print("=" * 60)
#         print("Feature Engineering Completed Successfully")
#         print(f"Training Samples : {len(X_train)}")
#         print(f"Testing Samples  : {len(X_test)}")
#         print("=" * 60)
#         return X_train, X_test, y_train, y_test


# if __name__ == "__main__":
#     feature_engineering = FeatureEngineering()
#     feature_engineering.engineer_features()


import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    def __init__(self):
        self.input_path = "data/processed/processed_data.csv"
        self.output_dir = "data/features"
        self.parquet_path = "feature_repo/feature_repo/data/features.parquet"

    def engineer_features(self):
        # Load processed dataset
        df = pd.read_csv(self.input_path)
        print(df.columns.tolist())

        # --------------------------------------------------
        # Create Parquet File for Feast
        # --------------------------------------------------

        feast_df = df.copy()

        # Create property_id if it does not exist
        if "property_id" not in feast_df.columns:
            feast_df["property_id"] = range(1, len(feast_df) + 1)

        # Select only required columns
        feast_df = feast_df[
            [
                "property_id",
        "property_type",
        "location",
        "city",
        "province_name",
        "latitude",
        "longitude",
        "baths",
        "purpose",
        "bedrooms",
        "Area Type",
        "Area Size",
        "Area Category"
            ]
        ]

        # Add timestamp columns
        feast_df["event_timestamp"] = datetime.now()
        feast_df["created_timestamp"] = datetime.now()

        # Save parquet file
        feast_df.to_parquet(self.parquet_path, index=False)

        print("✓ Feast parquet file created successfully.")

        # --------------------------------------------------
        # Feature Engineering for Model Training
        # --------------------------------------------------

        # Separate features and target
        X = df.drop("price", axis=1)
        y = df["price"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Scale numeric columns
        scaler = StandardScaler()

        numeric_columns = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns

        X_train[numeric_columns] = scaler.fit_transform(
            X_train[numeric_columns]
        )

        X_test[numeric_columns] = scaler.transform(
            X_test[numeric_columns]
        )

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Save train-test datasets
        X_train.to_csv(
            f"{self.output_dir}/X_train.csv",
            index=False
        )

        X_test.to_csv(
            f"{self.output_dir}/X_test.csv",
            index=False
        )

        y_train.to_csv(
            f"{self.output_dir}/y_train.csv",
            index=False
        )

        y_test.to_csv(
            f"{self.output_dir}/y_test.csv",
            index=False
        )

        print("=" * 60)
        print("Feature Engineering Completed Successfully")
        print("Feast Parquet : data/features.parquet")
        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")
        print("=" * 60)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    feature_engineering = FeatureEngineering()
    feature_engineering.engineer_features()