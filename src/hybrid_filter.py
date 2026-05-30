from typing import Dict, Any

from src.regex_filter import RegexToxicFilter
from src.bert_filter import BertToxicFilter


class HybridToxicFilter:
    def __init__(self, regex_bonus: float = 0.10, threshold: float = 0.5):
        self.regex_filter = RegexToxicFilter()
        self.bert_filter = BertToxicFilter()
        self.regex_bonus = regex_bonus
        self.threshold = threshold

    def predict_one(self, text: str) -> Dict[str, Any]:
        regex_result = self.regex_filter.predict_one(text)
        bert_result = self.bert_filter.predict_one(text)

        bert_score = bert_result["score"]

        if bert_result["label"].lower() in ["non-toxic", "label_0", "0"]:
            toxic_score = 1.0 - bert_score
        else:
            toxic_score = bert_score

        adjusted_score = toxic_score

        if regex_result["is_toxic"] == 1:
            adjusted_score = min(1.0, toxic_score + self.regex_bonus)

        is_toxic = int(adjusted_score >= self.threshold)

        return {
            "is_toxic": is_toxic,
            "score": adjusted_score,
            "bert_score": toxic_score,
            "regex_matches": regex_result["matches"],
            "method": "soft_hybrid"
        }

    def predict_batch(self, texts):
        return [self.predict_one(text) for text in texts]