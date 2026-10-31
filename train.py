"""Train, compare and save the spam classifier.

Usage (from the project root):
    python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocess import clean_text, load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam.csv"
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
FIG_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SEED = 42


def make_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            ("clf", clf),
        ]
    )


def candidate_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=10, max_iter=1000, class_weight="balanced", random_state=SEED
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def plot_class_distribution(df):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="label", data=df, hue="label", palette=["#4C9F70", "#D1495B"], legend=False)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Class distribution (after removing duplicates)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()


def plot_length_distribution(df):
    plt.figure(figsize=(6, 4))
    for label, color in [("ham", "#4C9F70"), ("spam", "#D1495B")]:
        sns.histplot(df[df.label == label]["text"].str.len(), bins=40, alpha=0.6,
                     color=color, label=label, stat="density")
    plt.xlabel("Message length (characters)")
    plt.title("Message length: ham vs spam")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "message_length.png", dpi=150)
    plt.close()


def plot_model_comparison(cv_scores: dict):
    names = list(cv_scores)
    means = [cv_scores[n].mean() for n in names]
    stds = [cv_scores[n].std() for n in names]
    plt.figure(figsize=(7, 4))
    plt.barh(names, means, xerr=stds, color="#3A6EA5")
    plt.xlim(0.8, 1.0)
    plt.xlabel("Cross-validated F1 (spam class)")
    plt.title("Model comparison (5-fold CV on training set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc(y_true, proba, auc):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#D1495B", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close()


def plot_top_words(pipeline, n=15):
    """Top spam / ham words from the logistic-regression coefficients."""
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    words = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_spam = np.argsort(coefs)[-n:]
    top_ham = np.argsort(coefs)[:n]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(words[top_spam], coefs[top_spam], color="#D1495B")
    axes[0].set_title("Top spam indicators")
    axes[1].barh(words[top_ham], -coefs[top_ham], color="#4C9F70")
    axes[1].set_title("Top ham indicators")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_words.png", dpi=150)
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} unique messages | spam: {df.target.sum()} | ham: {(1 - df.target).sum()}")
    plot_class_distribution(df)
    plot_length_distribution(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2, stratify=df["target"], random_state=SEED
    )

    # 1) Compare candidate models with cross-validation (training data only)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, clf in candidate_models().items():
        scores = cross_val_score(make_pipeline(clf), X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        cv_scores[name] = scores
        print(f"{name:<20} CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    plot_model_comparison(cv_scores)

    best_name = max(cv_scores, key=lambda n: cv_scores[n].mean())
    print(f"\nBest model by CV F1: {best_name}")

    # 2) Fit the best model on the full training set and evaluate on the held-out test set
    best_model = make_pipeline(candidate_models()[best_name])
    best_model.fit(X_train, y_train)
    test_metrics = evaluate(best_model, X_test, y_test)
    y_pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]

    print("\nTest set results")
    for k, v in test_metrics.items():
        print(f"  {k:<10}{v:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    plot_confusion(y_test, y_pred)
    plot_roc(y_test, proba, test_metrics["roc_auc"])

    # 3) Interpretability: fit a logistic regression just to read off top words
    lr_for_words = make_pipeline(
        LogisticRegression(C=10, max_iter=1000, class_weight="balanced", random_state=SEED)
    ).fit(X_train, y_train)
    plot_top_words(lr_for_words)

    # 4) Save model + metrics
    joblib.dump(best_model, MODEL_PATH)
    results = {
        "dataset": {
            "unique_messages": int(len(df)),
            "spam": int(df.target.sum()),
            "ham": int((1 - df.target).sum()),
        },
        "cv_f1": {n: {"mean": float(s.mean()), "std": float(s.std())} for n, s in cv_scores.items()},
        "best_model": best_name,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved model -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics -> {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
