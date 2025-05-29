import pandas as pd
import re

def clean_text(text):
    """
    A simple function to clean text data.
    - Lowercase
    - Remove HTML tags
    - Remove punctuation (simple version)
    - Remove extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def main():
    # --- 1. Collect (Simulated by reading a local CSV) ---
    try:
        raw_df = pd.read_csv("sample_data_raw.csv")
        print("Successfully loaded raw data.")
    except FileNotFoundError:
        print("Error: sample_data_raw.csv not found. Please create it.")
        return

    # --- 2. Clean ---
    cleaned_texts = []
    for text_content in raw_df['text']: # Changed raw_df.text to raw_df['text'] for clarity
        cleaned_texts.append(clean_text(text_content))
    
    cleaned_df = pd.DataFrame({'cleaned_text': cleaned_texts})
    
    # --- 3. Store/Output (Simulated by saving to a new CSV) ---
    cleaned_df.to_csv("sample_data_cleaned.csv", index=False)
    print(f"Successfully cleaned data and saved to sample_data_cleaned.csv")
    print("\nCleaned Data Head:")
    print(cleaned_df.head())

if __name__ == "__main__":
    main()