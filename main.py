"""
Main workflow script for IMDb sentiment analysis.
Orchestrates data loading, cleaning, feature extraction, and baseline modeling.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import load_imdb_data, preprocess_dataset
from feature_extraction import create_tfidf_features, save_tfidf_features
from baseline_model import train_baseline_model, evaluate_model


def create_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test splits.
    
    Args:
        df: DataFrame with cleaned reviews
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (from remaining after test)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*60)
    print("Creating train/validation/test splits...")
    print("="*60)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['review_clean'],
        df['sentiment'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['sentiment']
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """
    Main workflow function.
    """
    print("="*60)
    print("IMDb Sentiment Analysis - Baseline Workflow")
    print("="*60)
    
    # Configuration
    DATA_PATH = 'data/imdb_dataset.zip'  # Zip archive containing IMDB Dataset.csv
    OUTPUT_DIR = 'outputs'
    CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, 'clean_reviews.csv')
    TFIDF_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'tfidf_features.pkl')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Load IMDb dataset
    print("\n" + "="*60)
    print("Step 1: Loading IMDb Dataset")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please ensure the zip archive 'imdb_dataset.zip' is in the 'data' directory.")
        return
    
    df = load_imdb_data(DATA_PATH)
    
    # Step 2: Clean text data
    print("\n" + "="*60)
    print("Step 2: Cleaning Text Data")
    print("="*60)
    df_clean = preprocess_dataset(df, remove_stopwords=True)
    
    # Save cleaned dataset
    print(f"\nSaving cleaned dataset to {CLEAN_DATA_PATH}...")
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)
    print("Cleaned dataset saved successfully")
    
    # Step 3: Create train/validation/test splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(
        df_clean, 
        test_size=0.2, 
        val_size=0.1, 
        random_state=42
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Step 4: Build TF-IDF feature matrices
    print("\n" + "="*60)
    print("Step 4: Building TF-IDF Feature Matrices")
    print("="*60)
    tfidf_data = create_tfidf_features(
        X_train, X_val, X_test,
        unigram=True,
        bigram=True,
        max_features=None  # Use all features (can set to limit, e.g., 10000)
    )
    
    # Save TF-IDF features
    save_tfidf_features(tfidf_data, TFIDF_FEATURES_PATH)
    
    # Step 5: Train baseline model
    print("\n" + "="*60)
    print("Step 5: Training Baseline Model (TF-IDF + Logistic Regression)")
    print("="*60)
    model = train_baseline_model(
        tfidf_data['X_train'],
        y_train_encoded,
        tfidf_data['X_val'],
        y_val_encoded,
        random_state=42
    )
    
    # Step 6: Evaluate on test set
    print("\n" + "="*60)
    print("Step 6: Evaluating on Test Set")
    print("="*60)
    results = evaluate_model(
        model,
        tfidf_data['X_test'],
        y_test_encoded
    )
    
    # Save results summary
    results_summary = {
        'test_accuracy': results['accuracy'],
        'test_f1_score': results['f1_score']
    }
    
    print("\n" + "="*60)
    print("Workflow Complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  - Clean dataset: {CLEAN_DATA_PATH}")
    print(f"  - TF-IDF features: {TFIDF_FEATURES_PATH}")
    print(f"\nBaseline Results:")
    print(f"  - Test Accuracy: {results_summary['test_accuracy']:.4f}")
    print(f"  - Test F1 Score: {results_summary['test_f1_score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

