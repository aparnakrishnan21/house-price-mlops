# import joblib
# import pandas as pd
# from fastapi import FastAPI

# from api.schemas import HouseData

# app = FastAPI(
#     title="House Price Prediction API",
#     version="1.0"
# )

# model = joblib.load("models/best_model.pkl")


# @app.get("/")
# def home():
#     return {"message": "House Price Prediction API is Running"}


# @app.post("/predict")
# def predict(data: HouseData):

#     input_df = pd.DataFrame([{
#         "property_type": data.property_type,
#         "location": data.location,
#         "city": data.city,
#         "province_name": data.province_name,
#         "latitude": data.latitude,
#         "longitude": data.longitude,
#         "baths": data.baths,
#         "purpose": data.purpose,
#         "bedrooms": data.bedrooms,
#         "Area Type": data.Area_Type,
#         "Area Size": data.Area_Size,
#         "Area Category": data.Area_Category
#     }])

#     prediction = model.predict(input_df)

#     return {
#         "Predicted Price": float(prediction[0])
#     }

import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from feast import FeatureStore
import os
import logging
from monitoring.system_health import get_system_health
#from monitoring.system_metric import REQUEST_COUNT
# from fastapi.responses import Response 
# from prometheus_client import generate_latest 
# Create logs folder if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
#from api.schemas import HouseData

app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)

model = joblib.load("models/best_model.pkl")
# Load Feast Feature Store
store = FeatureStore(repo_path="feature_repo/feature_repo")



@app.get("/")
def home():
    return {"message": "House Price Prediction API is Running"}

# @app.get("/metrics")
# def metrics():
#     return Response(
#         generate_latest(),
#         media_type="text/plain"
#     )

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "system": get_system_health()
    }


@app.post("/predict/{property_id}")
def predict(property_id: int):
    #REQUEST_COUNT.inc()
    # Fetch features from Feast
    feature_vector = store.get_online_features(
        features=[
            "house_features:property_type",
            "house_features:location",
            "house_features:city",
            "house_features:province_name",
            "house_features:latitude",
            "house_features:longitude",
            "house_features:baths",
            "house_features:purpose",
            "house_features:bedrooms",
            "house_features:Area Type",
            "house_features:Area Size",
            "house_features:Area Category",
        ],
        entity_rows=[{"property_id": property_id}],
    ).to_dict()

     # Check if property exists
    if feature_vector["bedrooms"][0] is None:
        logging.warning(
        f"Prediction failed. Property ID {property_id} not found."
    )
        raise HTTPException(
            status_code=404,
            detail=f"Property ID {property_id} not found."
        )

    # Prepare model input
    input_df = pd.DataFrame([{
        "property_type": feature_vector["property_type"][0],
        "location": feature_vector["location"][0],
        "city": feature_vector["city"][0],
        "province_name": feature_vector["province_name"][0],
        "latitude": feature_vector["latitude"][0],
        "longitude": feature_vector["longitude"][0],
        "baths": feature_vector["baths"][0],
        "purpose": feature_vector["purpose"][0],
        "bedrooms": feature_vector["bedrooms"][0],
        "Area Type": feature_vector["Area Type"][0],
        "Area Size": feature_vector["Area Size"][0],
        "Area Category": feature_vector["Area Category"][0],
    }])

    # Predict
    prediction = model.predict(input_df)
    logging.info(
    f"Property ID={property_id}, "
    f"Prediction={float(prediction[0])}, "
    f"Features={input_df.to_dict(orient='records')[0]}"
)

    return {
        "property_id": property_id,
        "predicted_price": float(prediction[0])
    }

