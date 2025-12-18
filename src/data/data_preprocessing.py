import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
from from_root import from_root
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text: str) -> str:
    """  Applying text preprocessing to a specific column. """
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        # Remove URLs
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

        return text
        
    except Exception as e:
        logging.error('Unexpected error during text preprocessing: %s', e)
        raise

def preprocess_dataframe(df, col) -> pd.DataFrame:
    """  Preprocess a DataFrame. """
    try:
        df[col] = df[col].apply(preprocess_text)
        df = df.dropna()
        logging.info("Data pre-processing completed")
        return df
        
    except Exception as e:
        logging.error('Unexpected error during text preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        logging.info('Processed train and test data saved to %s', interim_data_path)

    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise   

def main():
    try:
        data_path = os.path.join(from_root(), 'data')

        # Fetch the data from data/raw
        train_data = pd.read_csv(f'{data_path}/raw/train.csv')
        test_data  = pd.read_csv(f'{data_path}/raw/test.csv')
        logging.info('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data  = preprocess_dataframe(test_data, 'review')

        # Store the data inside data/interim
        save_data(train_processed_data, test_processed_data, data_path=data_path)
        
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()