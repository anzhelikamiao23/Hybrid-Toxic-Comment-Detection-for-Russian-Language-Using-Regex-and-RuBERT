import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    TEXT_COL,
    LABEL_COL,
    RANDOM_STATE,
    TEST_SIZE
)

from src.preprocess import clean_text
from src.regex_filter import RegexToxicFilter


df = pd.read_csv(DATA_PATH)

df[TEXT_COL] = (
    df[TEXT_COL]
    .astype(str)
    .apply(clean_text)
)

_, test = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[LABEL_COL]
)

model = RegexToxicFilter()

preds = model.predict_batch(
    test[TEXT_COL].tolist()
)

test = test.copy()

test["pred"] = [
    x["is_toxic"]
    for x in preds
]

test["regex_matches"] = [
    ", ".join(x["matches"])
    for x in preds
]

# FALSE POSITIVE
fp = test[
    (test[LABEL_COL] == 0)
    &
    (test["pred"] == 1)
]

fp = fp[
    [
        TEXT_COL,
        LABEL_COL,
        "pred",
        "regex_matches"
    ]
]

fp.to_csv(
    "reports/regex_false_positive.csv",
    index=False
)

print("FP:", len(fp))

print(fp.head(20))