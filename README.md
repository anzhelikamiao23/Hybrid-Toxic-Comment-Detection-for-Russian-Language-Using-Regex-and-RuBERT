# Hybrid Russian Toxicity Detection

This project implements and evaluates a hybrid system for Russian toxic comment detection.

The system combines:

* rule-based regex filtering for explicit profanity;
* RuBERT-based toxic comment classification;
* a hybrid decision layer combining regex and RuBERT outputs.

## Task

The task is binary classification of Russian comments:

* `0` — non-toxic
* `1` — toxic

## Dataset

The main benchmark dataset is the **Russian Language Toxic Comments Dataset**, which contains comments collected from Pikabu and 2ch.

The project also includes evaluation on real-world bank customer support chat logs. These logs are not included in the repository due to privacy and confidentiality restrictions.

## Models

The following approaches are compared:

1. **Regex baseline** — detects explicit profanity using manually designed regular expressions.
2. **RuBERT classifier** — uses the pretrained `sismetanin/rubert-toxic-pikabu-2ch` model.
3. **Hybrid Regex + RuBERT** — combines regex matches with the RuBERT toxicity score.

## Results

### Benchmark Dataset

| Model  | Accuracy | Precision | Recall |    F1 |
| ------ | -------: | --------: | -----: | ----: |
| Regex  |    0.635 |     0.454 |  0.436 | 0.445 |
| RuBERT |    0.956 |     0.935 |  0.934 | 0.934 |
| Hybrid |    0.955 |     0.932 |  0.935 | 0.933 |

### Real-World Support Logs

| Model  | Accuracy | Precision | Recall |    F1 |
| ------ | -------: | --------: | -----: | ----: |
| Regex  |    0.622 |     0.941 |  0.260 | 0.407 |
| RuBERT |    0.797 |     0.792 |  0.805 | 0.799 |
| Hybrid |    0.797 |     0.792 |  0.805 | 0.799 |

## Project Structure

```text
src/
├── config.py
├── preprocess.py
├── regex_filter.py
├── bert_filter.py
├── hybrid_filter.py
├── evaluate.py
├── run_regex.py
├── run_experiments.py
└── analyze_regex_fp.py

notebooks/
└── test_logs_pipeline.ipynb

reports/
└── experiment outputs
```

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Run the regex baseline:

```bash
python -m src.run_regex
```

Run full experiments:

```bash
python -m src.run_experiments
```

Analyze regex false positives:

```bash
python -m src.analyze_regex_fp
```

## Notes

The real-world customer support logs are not published because they may contain confidential information from customer-operator conversations. The repository contains only code, reports and reproducible experiments on the public benchmark dataset.
