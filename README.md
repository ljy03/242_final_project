# IMDb Sentiment Analysis - Baseline Workflow

This project implements a complete workflow for IMDb sentiment analysis using TF-IDF features and Logistic Regression as a baseline model.

## Project Structure

```
242_final_project/
├── data/
│   └── imdb_dataset.zip          # Input dataset (zip archive containing IMDB Dataset.csv)
├── src/
│   ├── data_processing.py        # Data loading and text cleaning
│   ├── feature_extraction.py     # TF-IDF feature extraction
│   └── baseline_model.py         # Baseline model training and evaluation
├── outputs/                       # Generated outputs (not in repo, run main.py to create)
│   ├── clean_reviews.csv         # Cleaned dataset (generated)
│   └── tfidf_features.pkl         # TF-IDF feature matrices (generated)
├── main.py                       # Main workflow script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Workflow Steps

1. **Load IMDb Dataset**: Loads the dataset from zip archive (extracts CSV automatically)
2. **Clean Text Data**: 
   - Removes HTML tags
   - Removes punctuation
   - Removes stopwords
   - Converts to lowercase
3. **Create Splits**: Creates train/validation/test splits (70%/10%/20%)
4. **Build TF-IDF Features**: Creates unigram and bigram TF-IDF feature matrices
5. **Train Baseline Model**: Trains Logistic Regression with TF-IDF features
6. **Evaluate**: Evaluates model on test set and outputs accuracy/F1 score

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the IMDb dataset zip archive `imdb_dataset.zip` is in the `data/` directory.

3. **Generate outputs**: Run `python main.py` to generate the output files (they are not included in the repository).

   The zip archive should contain:
   - A CSV file (e.g., `IMDB Dataset.csv`) with:
     - A column with review text (named 'review', 'text', or 'comment')
     - A column with sentiment labels (named 'sentiment', 'label', or binary 0/1)

## Usage

**Important**: Output files (`outputs/clean_reviews.csv` and `outputs/tfidf_features.pkl`) are **not included** in this repository (they are excluded via `.gitignore` due to file size limits). You must run `main.py` first to generate these files.

Run the main workflow:
```bash
python main.py
```

This will generate all output files in the `outputs/` directory.

## Outputs

After running `main.py`, the following files will be generated in the `outputs/` directory:

- **clean_reviews.csv**: Cleaned dataset with original and cleaned review columns
- **tfidf_features.pkl**: Pickle file containing:
  - `X_train`: Training TF-IDF features
  - `X_val`: Validation TF-IDF features
  - `X_test`: Test TF-IDF features
  - `vectorizer`: Fitted TF-IDF vectorizer

**Note**: These output files are not tracked in git due to their large size (>100MB). Each team member should run `main.py` to generate them locally.

## Baseline Results

The script outputs:
- Validation accuracy and F1 score
- Test accuracy and F1 score
- Classification report

## Notes

- The script automatically downloads required NLTK data (punkt tokenizer and stopwords)
- Text cleaning removes HTML, URLs, emails, and punctuation
- TF-IDF uses both unigrams and bigrams by default
- The model uses stratified splits to maintain class distribution
  ## Classical ML Models on TF-IDF (IVO)

In addition to the baseline Logistic Regression model, we evaluate multiple classical machine learning models trained on TF-IDF features to assess the effectiveness of different modeling approaches for text sentiment classification.

### Models Evaluated
Using the TF-IDF feature matrices generated in the baseline workflow, the following models are trained and compared:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Random Forest

All models are evaluated using a consistent train/validation/test split.

### Evaluation Metrics
Model performance is assessed on the test set using:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- Training time

An overfitting analysis is also conducted by comparing training, validation, and test performance.

### Results Summary
Among all TF-IDF–based models, **Linear SVM** achieves the best overall performance, obtaining the highest test-set F1 score while maintaining stable generalization and low computational cost. Tree-based models such as Random Forest exhibit strong training performance but suffer from overfitting due to the high-dimensional and sparse nature of TF-IDF features.

### Outputs
The comparison results of TF-IDF models are summarized in:
- `outputs/tfidf_model_comparison.csv` (generated locally)

### Implementation
The implementation of TF-IDF classical model training and evaluation is provided in:


