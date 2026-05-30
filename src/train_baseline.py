import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import DATA_PATH, TEXT_COL, LABEL_COL, RANDOM_STATE, TEST_SIZE
from src.preprocess import clean_text
from src.evaluate import evaluate_predictions


def main():
    df = pd.read_csv(DATA_PATH)

    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(train_df[TEXT_COL], train_df[LABEL_COL])

    preds = model.predict(test_df[TEXT_COL])

    evaluate_predictions(
        test_df[LABEL_COL],
        preds,
        model_name="TF-IDF + Logistic Regression"
    )

    joblib.dump(model, "models/tfidf_logreg.pkl")


if __name__ == "__main__":
    main()