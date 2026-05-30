# Hybrid Russian Toxicity Detection

This project implements a two-level Russian toxic comment detector:

1. Rule-based regex filter for explicit profanity.
2. BERT-based toxic comment classifier.
3. Hybrid decision layer combining regex and BERT.

## Task

Binary classification of Russian comments:

- 0: non-toxic
- 1: toxic

## Dataset

The project uses Russian Language Toxic Comments dataset, which contains comments from Pikabu and 2ch.

## Models

We compare:

- Regex-only baseline
- RuBERT toxic classifier
- Hybrid Regex + RuBERT model

## How to run

Install dependencies:

```bash
pip install -r requirements.txt