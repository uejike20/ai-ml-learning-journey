# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
# ... (other imports: JSONResponse, torch, models, Image, os, logging, transform_image) ...
import os # Make sure os is imported
import logging
import torch
import torchvision.models as models
from fastapi.responses import JSONResponse
from PIL import Image # Ensure PIL Image is imported if not already
from preprocessing import transform_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using a fine-tuned ResNet18 model.",
    version="0.1.0"
)

# --- Configuration ---
# Define the expected model filename
MODEL_FILENAME = "cats_dogs_best_model.pth" # <<< CHANGE THIS if your model has a different name

# Path for when running LOCALLY (relative to api/main.py)
DEFAULT_MODEL_PATH_LOCAL = f"../models/{MODEL_FILENAME}"
# Path INSIDE THE DOCKER CONTAINER (relative to /app where main.py will be)
MODEL_PATH_IN_CONTAINER = f"model_weights/{MODEL_FILENAME}"

# Determine the model path based on an environment variable
# The Dockerfile will set RUNNING_IN_DOCKER=true
if os.getenv("RUNNING_IN_DOCKER") == "true":
    MODEL_PATH = MODEL_PATH_IN_CONTAINER
    logger.info(f"RUNNING_IN_DOCKER detected. Model path set to: {MODEL_PATH}")
else:
    MODEL_PATH = DEFAULT_MODEL_PATH_LOCAL
    logger.info(f"Not running in Docker (or RUNNING_IN_DOCKER not set). Model path set to: {MODEL_PATH}")


# CLASS_NAMES -  IMPORTANT: This MUST match your training order and classes.
CLASS_NAMES = ['cats', 'dogs'] # Example for Cats vs Dogs
# Example for products (MAKE SURE THIS MATCHES YOUR TRAINING):
# CLASS_NAMES = ['laptops', 'mugs', 'shoes', 'tshirts']

NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Expecting {NUM_CLASSES} classes: {CLASS_NAMES}")

# --- Load Model ---
model = None
try:
    model_abs_path = os.path.abspath(MODEL_PATH) # Get absolute path for clarity in logs
    logger.info(f"Attempting to load model from: {model_abs_path}")

    if not os.path.exists(MODEL_PATH): # Check existence using the determined MODEL_PATH
        logger.error(f"Model file not found at resolved path: {MODEL_PATH} (abs: {model_abs_path})")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded successfully from {MODEL_PATH}")

except Exception as e:
    logger.error(f"Error loading model: {e}", exc_info=True) # Log full traceback
    model = None

# ... (rest of your @app.get("/") and @app.post("/predict") endpoints remain the same) ...
@app.get("/")
async def root():
    return {"message": "Welcome to the Image Classification API! Use the /predict endpoint to upload an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        logger.error("Model is not loaded. Cannot predict.")
        raise HTTPException(status_code=503, detail="Model not loaded or error during loading.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (jpeg, png, etc.).")

    try:
        image_bytes = await file.read()
        tensor = transform_image(image_bytes)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            _, predicted_idx = torch.max(outputs, 1)
            
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = probabilities[predicted_idx.item()].item()
        
        all_scores = {CLASS_NAMES[i]: probabilities[i].item() for i in range(NUM_CLASSES)}
        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence_score:.4f}, All Scores: {all_scores}")

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence_score": confidence_score,
            "all_scores": all_scores
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")