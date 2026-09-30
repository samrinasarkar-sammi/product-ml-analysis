# Bengali Product Review Sentiment Classification

An individually-built NLP pipeline for classifying Bengali-language e-commerce product reviews as positive or negative, using a self-collected dataset and classical machine learning with TF-IDF n-gram features.

## Overview

This is a personal, individually-built project — not part of a published study. The dataset (300 Bengali product reviews, evenly split 150 positive / 150 negative, with 1–5 star ratings) was self-collected from an e-commerce platform review section during my own research time and has not been published elsewhere.

The pipeline: clean raw Bengali text, extract TF-IDF features at unigram, bigram, and trigram levels, and compare 7 classical ML classifiers on each.

## Dataset

- **Size:** 300 reviews (150 Positive, 150 Negative — perfectly balanced)
- **Fields:** Comment (raw Bengali text), Sentiment (0/Negative or 1/Positive), Stars (1–5)
- **Vocabulary:** 1,185 unique words total after cleaning; most frequent words per class are clearly polarized (Negative: না "no", বাজে "bad"; Positive: ভালো "good", ধন্যবাদ "thanks")
- **Review length:** 4–62 words, average 14 words

## Methodology

1. **Data cleaning:** regex-based filtering to retain only Bengali Unicode script (removing Latin characters, punctuation, and emojis)
2. **Feature extraction:** TF-IDF vectorization at three n-gram levels — unigram, bigram (1–2), and trigram (1–3)
3. **Train/test split:** 90/10 (270 train / 30 test)
4. **Models compared (7):** Logistic Regression, Decision Tree, Random Forest, Multinomial Naive Bayes, KNN, Linear SVM, RBF SVM
5. **Metrics:** Accuracy, Precision, Recall, F1 Score, evaluated separately for each n-gram feature set

## Results

### Unigram

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 96.67 | 100.00 | 92.31 | 96.00 |
| Decision Tree | 93.33 | 92.31 | 92.31 | 92.31 |
| Random Forest | 96.67 | 100.00 | 92.31 | 96.00 |
| **Multinomial Naive Bayes** | **100.00** | **100.00** | **100.00** | **100.00** |
| KNN | 90.00 | 85.71 | 92.31 | 88.89 |
| Linear SVM | 90.00 | 100.00 | 76.92 | 86.96 |
| RBF SVM | 90.00 | 100.00 | 76.92 | 86.96 |

### Bigram

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 96.67 | 100.00 | 92.31 | 96.00 |
| Decision Tree | 86.67 | 84.62 | 84.62 | 84.62 |
| Random Forest | 96.67 | 100.00 | 92.31 | 96.00 |
| **Multinomial Naive Bayes** | **100.00** | **100.00** | **100.00** | **100.00** |
| KNN | 90.00 | 91.67 | 84.62 | 88.00 |
| **Linear SVM** | **100.00** | **100.00** | **100.00** | **100.00** |
| RBF SVM | 90.00 | 100.00 | 76.92 | 86.96 |

### Trigram

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 96.67 | 100.00 | 92.31 | 96.00 |
| Decision Tree | 86.67 | 90.91 | 76.92 | 83.33 |
| Random Forest | 96.67 | 100.00 | 92.31 | 96.00 |
| **Multinomial Naive Bayes** | **100.00** | **100.00** | **100.00** | **100.00** |
| KNN | 86.67 | 84.62 | 84.62 | 84.62 |
| Linear SVM | 46.67 | 44.83 | 100.00 | 61.90 |
| **RBF SVM** | **100.00** | **100.00** | **100.00** | **100.00** |

## A Necessary Caveat: Interpreting the 100% Scores

Several models achieve a perfect 100% across every metric. **This should not be read as evidence of a flawless model** — with only 30 examples in the test set, getting every single one right is a real possibility on a small, clean dataset where the two classes use strongly distinct vocabulary (see the most-frequent-word lists above), not necessarily proof of a production-ready classifier. Two things support this "too clean to be fully trusted" reading rather than genuine robustness:

- **Test set size:** 30 examples is small enough that perfect accuracy carries wide uncertainty — a single unlucky example would already drop the score to 96.7%.
- **No cross-validation:** results come from a single train/test split, not averaged across multiple splits, so we cannot rule out this particular split being unusually easy.

The honest conclusion is that this dataset, in its current small and vocabulary-polarized form, is not a reliable indicator of how these models would perform on more ambiguous, larger-scale, real-world review data. Scaling the dataset and adding cross-validation would be the natural next step, not a nice-to-have.

## Tech Stack

Python, pandas, scikit-learn, matplotlib, seaborn

## Project Structure

```
├── ProductData_Code.ipynb    # Full analysis notebook
├── Product Data.csv          # Self-collected dataset (not published elsewhere)
└── README.md
```

## About Me

**Samrina Sarkar Sammi** — M2 Data Science & Network Intelligence student, Télécom SudParis

[LinkedIn](https://www.linkedin.com/in/samrina-sarkar-sammi-a8b716424/) · [GitHub](https://github.com/samrinasarkar-sammi) · samrinasarkar@gmail.com
