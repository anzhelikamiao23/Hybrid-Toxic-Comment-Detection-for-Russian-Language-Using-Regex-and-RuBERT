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
from src.evaluate import evaluate_predictions


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

texts = test[TEXT_COL].tolist()

model = RegexToxicFilter()

preds = model.predict_batch(texts)

y_pred = [
    x["is_toxic"]
    for x in preds
]

evaluate_predictions(
    test[LABEL_COL],
    y_pred,
    "Regex"
)