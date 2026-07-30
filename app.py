"""
RentIQ API
-----------
Serves the trained Linear Regression model as a /predict endpoint.

Run locally with:
    uvicorn app:app --reload

Then test at http://127.0.0.1:8000/docs (FastAPI's auto-generated
interactive docs - genuinely useful for testing without building
a frontend first).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="RentIQ API")

# Allow your frontend (running on a different port/domain) to call this API.
# For now this allows everything - lock this down to your real frontend
# domain once you deploy for real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model + scaler once, when the server starts
# (not on every request - that would be slow)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Defines exactly what a request must look like - FastAPI validates
# this automatically and returns a clear error if a field is missing
# or the wrong type, before your code even runs.
class PredictionRequest(BaseModel):
    surface_m2: float = Field(..., gt=0, description="Surface area in square meters")
    pieces: float = Field(..., ge=0, description="Total room count")
    bedrooms: float = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/")
def root():
    return {"message": "RentIQ API is running. See /docs for usage."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Build the feature vector in EXACTLY the same order used during training
    features = np.array([[
        request.surface_m2,
        request.pieces,
        request.bedrooms,
        request.bathrooms,
    ]])

    # Apply the SAME scaler fitted during training - never fit a new one here
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    return PredictionResponse(predicted_price=round(float(prediction), 2))