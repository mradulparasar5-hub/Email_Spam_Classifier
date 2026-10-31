from src.predict import load_model, predict_message
from src.preprocess import clean_text, load_dataset


def test_clean_text_replaces_patterns():
    out = clean_text("WIN £500 now! Visit http://x.com or call 09061701461")
    assert "moneytoken" in out
    assert "urltoken" in out
    assert "numbertoken" in out


def test_dataset_loads():
    df = load_dataset("data/spam.csv")
    assert {"label", "text", "target"} <= set(df.columns)
    assert df["target"].isin([0, 1]).all()


def test_model_predictions():
    model = load_model()
    spam = predict_message(model, "URGENT! You won a free prize. Call 09061701461 to claim your cash reward now")
    ham = predict_message(model, "Are we still on for dinner tonight? I'll be there at 7")
    assert spam["label"] == "spam"
    assert ham["label"] == "ham"
