from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline # Hugging Face pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Simple Sentiment Analysis API",
    description="An API that uses a pre-trained Hugging Face model for sentiment analysis.",
    version="0.1.0"
)

# --- Model Loading ---
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
    print("Sentiment analysis model loaded successfully.")
except Exception as e:
    print(f"Error loading sentiment analysis model: {e}")
    sentiment_analyzer = None

# --- Pydantic Models for Request and Response ---
class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API! Use the /predict endpoint to analyze text."}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    if sentiment_analyzer is None:
        # Using FastAPI's HTTPException for proper error responses
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Sentiment analysis model not available.")
        
    text_to_analyze = input_data.text
    
    try:
        result = sentiment_analyzer(text_to_analyze)
        
        if result and isinstance(result, list) and len(result) > 0:
            prediction = result[0]
            return SentimentResponse(
                text=text_to_analyze,
                label=prediction['label'],
                score=prediction['score']
            )
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Could not get a valid prediction from the model.")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")