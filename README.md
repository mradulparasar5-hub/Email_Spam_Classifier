# 📧 Email / SMS Spam Classifier

A complete machine learning project that classifies messages as **spam** or **ham** (legitimate) using TF-IDF features and classical ML models, with a Streamlit web app for live predictions.

## Results

Evaluated on a stratified 20% hold-out test set (1,034 messages):

| Metric | Score |
|---|---|
| Accuracy | 98.7% |
| Precision (spam) | 96.8% |
| Recall (spam) | 93.1% |
| F1 (spam) | 94.9% |
| ROC-AUC | 0.997 |

Best model: **Linear SVM** (probability-calibrated), selected by 5-fold cross-validated F1.

| Model | CV F1 (spam) |
|---|---|
| Linear SVM | 0.952 |
| Logistic Regression | 0.949 |
| Naive Bayes | 0.945 |
| Random Forest | 0.931 |

| Confusion matrix | ROC curve |
|---|---|
| ![cm](reports/figures/confusion_matrix.png) | ![roc](reports/figures/roc_curve.png) |

More plots are in [`reports/figures/`](reports/figures): class distribution, message length, model comparison, and top spam/ham words.

## Dataset

`data/spam.csv` has 5,572 labelled messages (`v1` = label, `v2` = text). It is imbalanced (~87% ham / 13% spam) and contains 403 duplicate messages. Duplicates are removed before splitting to avoid train/test leakage, leaving **5,169 unique messages** (4,516 ham, 653 spam). The file is `latin-1` encoded.

## Approach

1. **Cleaning** (`src/preprocess.py`): lower-casing; URLs, e-mail addresses, currency amounts and long numbers (phone numbers / short codes) are replaced by placeholder tokens so the model learns that they occur; punctuation removed.
2. **Features**: TF-IDF on unigrams + bigrams (`min_df=2`, sublinear TF, English stop-words removed).
3. **Models compared**: Multinomial Naive Bayes, Logistic Regression, Linear SVM (calibrated), Random Forest. Class imbalance is handled with `class_weight="balanced"` where supported.
4. **Selection**: 5-fold stratified CV on the training split, scored with F1 on the spam class (accuracy alone is misleading on imbalanced data).
5. **Final evaluation** on the untouched test split.

## Project structure

```
email-spam-classifier/
├── app.py                    # Streamlit web app
├── data/spam.csv             # dataset
├── models/spam_classifier.joblib   # trained pipeline (TF-IDF + classifier)
├── reports/
│   ├── metrics.json          # all metrics
│   └── figures/              # plots
├── src/
│   ├── preprocess.py         # cleaning + data loading
│   ├── train.py              # train / compare / evaluate / save
│   └── predict.py            # CLI + helper for predictions
├── tests/test_basic.py
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/<your-username>/email-spam-classifier.git
cd email-spam-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Train (optional, a trained model is already included):**
```bash
python -m src.train
```

**Predict from the command line:**
```bash
python -m src.predict "URGENT! You won a £1000 prize. Call 09061701461 now"
```

**Launch the web app:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
pytest
```

## Limitations and ideas for improvement

- The dataset is the classic English SMS Spam Collection, so performance on long, modern e-mails (HTML, phishing, newsletters) is not guaranteed; retraining on e-mail data would help.
- Recall on spam (93%) leaves room to grow; try threshold tuning, character n-grams, or transformer embeddings (e.g. DistilBERT).
- Add hyper-parameter search (`GridSearchCV`) and model explainability (LIME / SHAP).

## License

MIT
