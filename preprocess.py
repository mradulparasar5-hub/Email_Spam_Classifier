"""Text cleaning and dataset loading utilities."""
import re
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b\d{5,}\b")
CURRENCY_RE = re.compile(r"[£$€]\s?\d+[\d,.]*")
NON_ALNUM_RE = re.compile(r"[^a-z\s]")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case a message and replace noisy patterns with placeholder tokens.

    URLs, e-mail addresses, long numbers (phone numbers / short codes) and
    currency amounts are strong spam signals, so we keep the *fact* that they
    appeared (e.g. ``urltoken``) instead of the exact characters.
    """
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = CURRENCY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" numbertoken ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def load_dataset(path, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load spam.csv and return a tidy DataFrame with columns ``label``, ``text``, ``target``.

    The raw file is latin-1 encoded, has the columns ``v1`` (label) and ``v2``
    (message) plus a few almost-empty unnamed columns that are discarded.
    """
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]
    df = df.dropna()
    if drop_duplicates:
        # Duplicates across the train/test split would leak information.
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["target"] = (df["label"].str.lower() == "spam").astype(int)
    return df
