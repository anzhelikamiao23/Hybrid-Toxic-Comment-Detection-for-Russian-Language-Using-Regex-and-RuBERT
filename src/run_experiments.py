import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, TEXT_COL, LABEL_COL, RANDOM_STATE, TEST_SIZE
from src.preprocess import clean_text
from src.regex_filter import RegexToxicFilter
from src.bert_filter import BertToxicFilter
from src.hybrid_filter import HybridToxicFilter
from src.evaluate import evaluate_predictions


def main():
    df = pd.read_csv(DATA_PATH)

    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)

    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL]
    )

    texts = test_df[TEXT_COL].tolist()
    y_true = test_df[LABEL_COL].tolist()

    results = []

    regex_model = RegexToxicFilter()
    regex_outputs = regex_model.predict_batch(texts)
    regex_preds = [x["is_toxic"] for x in regex_outputs]
    results.append(evaluate_predictions(y_true, regex_preds, "Regex only"))

    bert_model = BertToxicFilter()
    bert_outputs = bert_model.predict_batch(texts)
    bert_preds = [x["is_toxic"] for x in bert_outputs]
    results.append(evaluate_predictions(y_true, bert_preds, "RuBERT toxic"))

    hybrid_model = HybridToxicFilter()
    hybrid_outputs = hybrid_model.predict_batch(texts)
    hybrid_preds = [x["is_toxic"] for x in hybrid_outputs]
    results.append(evaluate_predictions(y_true, hybrid_preds, "Regex + RuBERT"))

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/experiment_results.csv", index=False)

    test_df = test_df.copy()
    test_df["regex_pred"] = regex_preds
    test_df["bert_pred"] = bert_preds
    test_df["hybrid_pred"] = hybrid_preds
    test_df.to_csv("reports/test_predictions.csv", index=False)

    print("\nSaved reports:")
    print("reports/experiment_results.csv")
    print("reports/test_predictions.csv")


if __name__ == "__main__":
    main()