from typing import Dict, Any, List
from transformers import pipeline

from src.config import BERT_MODEL_NAME, TOXIC_THRESHOLD


class BertToxicFilter:
    def __init__(self, model_name: str = BERT_MODEL_NAME, threshold: float = TOXIC_THRESHOLD):
        self.threshold = threshold
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name
        )

    def predict_one(self, text: str) -> Dict[str, Any]:
        result = self.pipe(
            text,
            truncation=True,
            max_length=512
        )[0]

        label = result["label"]
        score = float(result["score"])

        toxic_by_label = label.lower() in ["toxic", "label_1", "1"]

        is_toxic = int(toxic_by_label and score >= self.threshold)

        return {
            "is_toxic": is_toxic,
            "score": score,
            "label": label,
            "method": "bert"
        }

    def predict_batch(self, texts: List[str]):
        return [self.predict_one(text) for text in texts]