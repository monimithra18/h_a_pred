from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pre-trained components
scaler = joblib.load("model/scaler.joblib")
pca = joblib.load("model/pca.joblib")
model = joblib.load("model/final_model_pca_lr.joblib")


# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Heart Attack Prediction API"}
    
# Input schema using Pydantic
class ModelInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict/")
async def predict(input_data: ModelInput):
    """
    Predict heart attack risk based on input features.
    """
    # Parse input data
    input_features = np.array([[
        input_data.age, input_data.sex, input_data.cp, input_data.trestbps,
        input_data.chol, input_data.fbs, input_data.restecg, input_data.thalach,
        input_data.exang, input_data.oldpeak, input_data.slope, input_data.ca,
        input_data.thal
    ]])

    # Preprocess input data
    scaled_data = scaler.transform(input_features)
    pca_data = pca.transform(scaled_data)

    # Predict using the loaded model
    prediction = model.predict(pca_data)
    probability = model.predict_proba(pca_data)

    # Return results
    return {
        "prediction": int(prediction[0]),
        "probability": probability[0].tolist()
    }
