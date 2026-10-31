"""Command-line / importable prediction helper.

Usage (from the project root):
    python -m src.predict "Congratulations! You won a free prize, click here"
    python -m src.predict            # interactive mode
"""
import sys
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_classifier.joblib"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_message(model, text: str) -> dict:
    proba = float(model.predict_proba([text])[0, 1])
    return {"label": "spam" if proba >= 0.5 else "ham", "spam_probability": proba}


def main():
    model = load_model()
    if len(sys.argv) > 1:
        messages = [" ".join(sys.argv[1:])]
    else:
        print("Type a message (empty line to quit):")
        messages = iter(lambda: input("> "), "")
    for msg in messages:
        r = predict_message(model, msg)
        print(f"{r['label'].upper():<5} (spam probability: {r['spam_probability']:.2%})")


if __name__ == "__main__":
    main()
