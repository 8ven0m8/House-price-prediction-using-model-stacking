from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load trained model (single XGB)
xgb = pickle.load(open("model.pkl", "rb"))

# EXACT columns from X_train (must match)
expected_cols = [
    "sqft_living", "sqft_lot", "floors", "view", "condition", "grade",
    "sqft_above", "yr_built", "yr_renovated", "lat", "long",
    "sqft_living15", "sqft_lot15", "year", "location_cluster",
    "living_density", "house_age", "zipcode_mean"
]

@app.get("/")
def home():
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict(data: dict):
    try:
        # convert input → DataFrame
        features = pd.DataFrame([data])

        # enforce same column order as training
        features = features[expected_cols]

        # single model prediction
        prediction = xgb.predict(features)

        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}