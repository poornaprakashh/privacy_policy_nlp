# Are Privacy Policies Designed to Confuse Users?

A text analysis project examining whether linguistic features of confusing privacy policies are systematic enough to be identified by a machine classifier.

---

## Project Overview

This project uses three complementary NLP methods — VADER sentiment analysis, LDA topic modelling, and DistilBERT-based supervised classification — to investigate whether privacy policies exhibit systematic linguistic patterns associated with deliberate obfuscation.

**Dataset:** https://www.kaggle.com/datasets/sonu1607/tosdr-terms-of-service-corpus
9,491 plain-text policy documents scraped from company websites (privacy policies, terms of service, cookie policies, acceptable use policies).

---

## Repository Structure

```
├── 01_preprocessing.ipynb        # Data loading, Flesch scoring, label construction
├── 02_vader.ipynb                # VADER sentence-level sentiment analysis
├── 03_LDA_Topic_Modelling.ipynb  # LDA topic modelling and chi-squared test
├── 04_BERT_Classifier.ipynb      # DistilBERT embeddings + LR and SVM classifiers
└── README.md
```

---

## How to Reproduce

### 1. Download the Data

Download the dataset from Kaggle:  
👉 https://www.kaggle.com/datasets/jkhan447/tosdr-privacy-policy-dataset

Extract the archive so that the `text/` folder (containing all `.txt` policy files) is accessible on your machine. Note the full path — you will need it in step 3.

### 2. Install Dependencies

All required packages are installed at the top of each notebook. However, you can also install them manually in advance:

```bash
pip install pandas numpy matplotlib seaborn tqdm textstat nltk vaderSentiment scipy gensim pyLDAvis transformers torch scikit-learn
```

For NLTK downloads (run once in Python):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Update File Paths

Each notebook contains file path variables at the top of the data-loading cell. Update these to match your local setup before running:

| Notebook | Variable to update | Points to |
|---|---|---|
| `01_preprocessing.ipynb` | `DATA_PATH` | Folder containing all `.txt` policy files |
| `02_vader.ipynb` | `SENTENCES_PATH`, `DOCS_PATH` | CSVs saved by notebook 01 |
| `03_LDA_Topic_Modelling.ipynb` | `SENTENCES_PATH`, `DOCS_PATH` | CSVs saved by notebook 01 |
| `04_BERT_Classifier.ipynb` | `SENTENCES_PATH`, `DOCS_PATH` | CSVs saved by notebook 01 |

### 4. Run Notebooks in Order

The notebooks must be run **in sequence** as each one depends on output files saved by the previous one:

```
01_preprocessing.ipynb  →  saves documents_labelled.csv and sentences_labelled.csv
02_vader.ipynb          →  loads both CSVs from step 01
03_LDA_Topic_Modelling  →  loads both CSVs from step 01
04_BERT_Classifier      →  loads both CSVs from step 01
```

---

## Runtime Expectations (CPU)

| Notebook | Estimated Runtime |
|---|---|
| 01 — Preprocessing | ~5–10 minutes |
| 02 — VADER | ~3–6 minutes (452,364 sentences) |
| 03 — LDA | ~10–15 minutes (lemmatisation + training) |
| 04 — BERT | ~15–30 minutes (embedding extraction) |

---

## Key Results

| Method | Key Finding |
|---|---|
| VADER | Confusing docs scored higher mean compound sentiment (0.1910 vs 0.1336, p < 0.001) |
| LDA (10 topics, c_v = 0.461) | Confusing docs cluster around data security and liability topics; Clear docs around tracking and communications |
| DistilBERT + Logistic Regression | 77.51% accuracy, 0.7774 F1 |
| DistilBERT + Linear SVM | 77.61% accuracy, 0.7805 F1 |

---

## Notes

- The raw dataset (~185 MB) is not included in this repository due to file size. Download it directly from the Kaggle link above.
- Text was truncated to 500 characters for DistilBERT embedding extraction due to CPU constraints.
- 27 documents with Flesch scores below −100 were removed as parsing failures before labelling.
- The middle 50% of documents by Flesch score were excluded to ensure clean class contrast (final labelled corpus: 4,732 documents, 50/50 split).
