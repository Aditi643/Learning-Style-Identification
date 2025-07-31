from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load("../Learning style prediction/hybrid_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class LearningStyleInput(BaseModel):
    features: list[float]  # Expecting a list of numerical inputs

# Prediction endpoint
@app.post("/predict/")
async def predict_learning_style(data: LearningStyleInput):
    # Convert input to NumPy array
    input_features = np.array(data.features).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(input_features)[0]

    # Map prediction to learning style
    label_mapping = {0: "Processing", 1: "Understanding", 2: "Input", 3: "Perception"}
    predicted_label = label_mapping.get(prediction, "Unknown")

    return {"prediction": predicted_label}

