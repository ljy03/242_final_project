"""
Feature extraction module for TF-IDF vectors.
Handles unigram and bigram TF-IDF feature extraction.
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(X_train, X_val, X_test, unigram=True, bigram=True, max_features=None):
    """
    Create TF-IDF feature matrices for train, validation, and test sets.
    
    Args:
        X_train: Training text data
        X_val: Validation text data
        X_test: Test text data
        unigram: Whether to include unigram features
        bigram: Whether to include bigram features
        max_features: Maximum number of features (None for all)
        
    Returns:
        Dictionary containing TF-IDF matrices and vectorizer
    """
    print("Creating TF-IDF features...")
    
    # Determine ngram range
    if unigram and bigram:
        ngram_range = (1, 2)
        print("  - Using unigram and bigram features")
    elif unigram:
        ngram_range = (1, 1)
        print("  - Using unigram features only")
    elif bigram:
        ngram_range = (2, 2)
        print("  - Using bigram features only")
    else:
        raise ValueError("At least one of unigram or bigram must be True")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Fit on training data and transform all sets
    print("  - Fitting vectorizer on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    print("  - Transforming validation and test data...")
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  - Feature matrix shape (train): {X_train_tfidf.shape}")
    print(f"  - Feature matrix shape (val): {X_val_tfidf.shape}")
    print(f"  - Feature matrix shape (test): {X_test_tfidf.shape}")
    print(f"  - Total vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return {
        'X_train': X_train_tfidf,
        'X_val': X_val_tfidf,
        'X_test': X_test_tfidf,
        'vectorizer': vectorizer
    }


def save_tfidf_features(tfidf_data, output_path):
    """
    Save TF-IDF features to pickle file.
    
    Args:
        tfidf_data: Dictionary containing TF-IDF matrices and vectorizer
        output_path: Path to save the pickle file
    """
    print(f"Saving TF-IDF features to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(tfidf_data, f)
    print("TF-IDF features saved successfully")


def load_tfidf_features(input_path):
    """
    Load TF-IDF features from pickle file.
    
    Args:
        input_path: Path to the pickle file
        
    Returns:
        Dictionary containing TF-IDF matrices and vectorizer
    """
    print(f"Loading TF-IDF features from {input_path}...")
    with open(input_path, 'rb') as f:
        tfidf_data = pickle.load(f)
    return tfidf_data

