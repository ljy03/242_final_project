"""
Baseline model module.
Implements TF-IDF + Logistic Regression baseline.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np


def train_baseline_model(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train baseline Logistic Regression model with TF-IDF features.
    
    Args:
        X_train: Training TF-IDF features
        y_train: Training labels
        X_val: Validation TF-IDF features
        y_val: Validation labels
        random_state: Random state for reproducibility
        
    Returns:
        Trained model
    """
    print("Training baseline Logistic Regression model...")
    
    # Create and train model
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='liblinear',  # Good for sparse data
        C=1.0
    )
    
    print("  - Fitting model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("  - Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"  - Validation Accuracy: {val_accuracy:.4f}")
    print(f"  - Validation F1 Score: {val_f1:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test TF-IDF features
        y_test: Test labels
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    print("Evaluating model on test set...")
    
    y_test_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_test_pred
    }

