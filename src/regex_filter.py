import re
from typing import Dict, Any


PROFANITY_PATTERNS = [
    r"\bх[уy][йеюяи]?\b",
    r"\bп[иi]зд[а-я]*\b",
    r"\bеб[а-я]*\b",
    r"\bёб[а-я]*\b",
    r"\bбл[яy][дт]?[а-я]*\b",
    r"\bсука\b",
    r"\bмраз[а-я]*\b",
    r"\bгандон[а-я]*\b",
    r"\bдолбо[её]б[а-я]*\b",
    r"\bмудак[а-я]*\b",
]

OBFUSCATED_PATTERNS = [
    r"\bх[\W_]*у[\W_]*й\b",

    r"\bп[\W_]*и[\W_]*з[\W_]*д[а-я]*\b",

    r"\b[её][\W_]*б[\W_]*а[\W_]*[тт]\b",

    r"\b[её][\W_]*б[\W_]*л[\W_]*я\b",
]


class RegexToxicFilter:
    def __init__(self):
        self.patterns = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in PROFANITY_PATTERNS + OBFUSCATED_PATTERNS
        ]

    def _normalize_match(self, match) -> str:
        if isinstance(match, tuple):
            match = "".join(match)

        return str(match)

    def _is_good_match(self, match: str) -> bool:
        compact = (
            match
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("*", "")
            .replace(".", "")
        )

        return len(compact) >= 3

    def predict_one(self, text: str) -> Dict[str, Any]:
        matches = []

        for pattern in self.patterns:
            found = pattern.findall(text)

            found = [
                self._normalize_match(x)
                for x in found
            ]

            found = [
                x
                for x in found
                if self._is_good_match(x)
            ]

            if found:
                matches.extend(found)

        is_toxic = len(matches) > 0

        return {
            "is_toxic": int(is_toxic),
            "score": 1.0 if is_toxic else 0.0,
            "matches": matches,
            "method": "regex"
        }

    def predict_batch(self, texts):
        return [self.predict_one(text) for text in texts]