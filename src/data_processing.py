"""
Data processing module for IMDb dataset.
Handles loading, cleaning, and preprocessing of text data.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import string
import zipfile
import io
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback to old punkt if punkt_tab not available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def load_imdb_data(data_path):
    """
    Load IMDb dataset from CSV file or zip archive.
    
    Args:
        data_path: Path to the IMDb dataset CSV file or zip archive containing CSV
        
    Returns:
        DataFrame with 'review' and 'sentiment' columns
    """
    print(f"Loading data from {data_path}...")
    
    # Check if it's a zip file
    if data_path.endswith('.zip'):
        print("  - Detected zip archive, extracting CSV...")
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            # Find CSV file in the zip
            csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError(f"No CSV file found in zip archive: {data_path}")
            
            # Use the first CSV file found (or 'IMDB Dataset.csv' if it exists)
            csv_file = 'IMDB Dataset.csv' if 'IMDB Dataset.csv' in csv_files else csv_files[0]
            print(f"  - Reading CSV from zip: {csv_file}")
            
            # Read CSV from zip
            with zip_ref.open(csv_file) as csv_file_obj:
                df = pd.read_csv(io.TextIOWrapper(csv_file_obj, encoding='utf-8'))
    else:
        # Regular CSV file
        df = pd.read_csv(data_path)
    
    # Standardize column names
    if 'review' in df.columns or 'text' in df.columns:
        review_col = 'review' if 'review' in df.columns else 'text'
        df = df.rename(columns={review_col: 'review'})
    elif 'comment' in df.columns:
        df = df.rename(columns={'comment': 'review'})
    else:
        # Assume first text column is review
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            df = df.rename(columns={text_cols[0]: 'review'})
    
    if 'sentiment' in df.columns:
        pass
    elif 'label' in df.columns:
        df = df.rename(columns={'label': 'sentiment'})
    elif 'rating' in df.columns:
        # Convert rating to sentiment (assuming 1-5 scale, <=2 negative, >=4 positive)
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
    else:
        # Assume binary classification: 0/1 or negative/positive
        label_cols = df.select_dtypes(include=['int64', 'int32']).columns
        if len(label_cols) > 0:
            df['sentiment'] = df[label_cols[0]].map({0: 'negative', 1: 'positive'})
        else:
            raise ValueError("Could not identify sentiment column. Please ensure dataset has review and sentiment columns.")
    
    print(f"Loaded {len(df)} reviews")
    return df[['review', 'sentiment']]


def remove_html_tags(text):
    """Remove HTML tags from text."""
    if pd.isna(text):
        return ""
    soup = BeautifulSoup(str(text), 'html.parser')
    return soup.get_text()


def clean_text(text, remove_stopwords=True):
    """
    Clean text by removing HTML, punctuation, and optionally stopwords.
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove HTML tags
    text = remove_html_tags(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        text = ' '.join(tokens)
    
    return text


def preprocess_dataset(df, remove_stopwords=True):
    """
    Preprocess the entire dataset.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        DataFrame with cleaned 'review' column
    """
    print("Cleaning text data...")
    print("  - Removing HTML tags...")
    print("  - Removing punctuation...")
    if remove_stopwords:
        print("  - Removing stopwords...")
    
    df_clean = df.copy()
    df_clean['review_clean'] = df_clean['review'].apply(
        lambda x: clean_text(x, remove_stopwords=remove_stopwords)
    )
    
    # Remove empty reviews after cleaning
    initial_count = len(df_clean)
    df_clean = df_clean[df_clean['review_clean'].str.len() > 0]
    removed_count = initial_count - len(df_clean)
    
    if removed_count > 0:
        print(f"  - Removed {removed_count} empty reviews after cleaning")
    
    print(f"Cleaned {len(df_clean)} reviews")
    return df_clean

