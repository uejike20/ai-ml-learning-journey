# Simple ML API Project: Sentiment Analysis

This project demonstrates a basic "data pipeline" and a FastAPI service for sentiment analysis using a pre-trained Hugging Face model.

## Project Structure

## Features

*   **Data Pipeline (`data_pipeline.py`):**
    *   Reads data from `sample_data_raw.csv`.
    *   Performs basic text cleaning (lowercase, remove HTML, remove punctuation, normalize whitespace).
    *   Saves the cleaned data to `sample_data_cleaned.csv`.
*   **API (`api.py`):**
    *   Uses FastAPI to create a web service.
    *   Loads a pre-trained sentiment analysis model from Hugging Face Transformers (`pipeline("sentiment-analysis")`).
    *   Provides a `/predict` endpoint that accepts JSON with a "text" field and returns the sentiment label (e.g., POSITIVE/NEGATIVE) and a confidence score.
    *   Includes interactive API documentation via Swagger UI (`/docs`) and ReDoc (`/redoc`).

## Setup and Usage

### 1. Prerequisites

*   Python 3.7+
*   `pip` (Python package installer)
*   (Optional) Docker Desktop

### 2. Local Setup

**a. Create a Virtual Environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate