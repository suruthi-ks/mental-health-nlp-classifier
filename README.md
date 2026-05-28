# 🧠 Mental Health Text Classification using NLP

> Automatically detecting mental health conditions from Reddit posts using classical NLP and machine learning — Naive Bayes, Logistic Regression, and Support Vector Machine.

---

## 📌 Project Overview

Mental health struggles are often expressed through informal, emotionally-charged language on social media — but rarely in clinical terms. This project builds a **7-class text classification system** trained on 52,681 real Reddit posts to automatically detect mental health conditions from raw text.

The system classifies posts into:
`Normal` · `Depression` · `Suicidal` · `Anxiety` · `Bipolar` · `Stress` · `Personality Disorder`

This was completed as an individual assignment for **IBM3201 — Data Mining & Predictive Analytics** (Jan 2026, BCSI, New INTI International University).

---

## 📊 Key Results

| Model | Accuracy | Weighted F1-Score | Training Time |
|---|---|---|---|
| Naive Bayes | 68.7% | 66.9% | 0.11s |
| Logistic Regression | **76.3%** | **76.3%** | 19.71s |
| Support Vector Machine | 75.4% | 75.2% | 6.06s |

**Best model: Logistic Regression** (76.3% accuracy, 76.3% weighted F1-Score)

All models achieved highest accuracy on the `Normal` class (Logistic Regression: 90%, SVM: 93%) and struggled most with `Stress` and `Personality Disorder` — reflecting genuine linguistic overlap between conditions.

---

## 🗂️ Repository Structure

```
mental-health-nlp/
│
├── notebooks/
│   └── mental_health_text_classification.ipynb   # Full pipeline: EDA → preprocessing → modelling → evaluation
│
├── reports/
│   └── NLP_MentalHealth_Report.pdf               # Full written report (20 pages)
│
├── assets/
│   ├── v1_class_distribution.png                 # Bar chart: dataset label distribution
│   ├── v2_pie_chart.png                          # Pie chart: percentage breakdown
│   ├── v3_text_length.png                        # Histogram: post text lengths
│   ├── v4_avg_length.png                         # Bar chart: avg post length by category
│   ├── v5_wordcloud.png                          # Word cloud: most frequent terms
│   ├── v6_confusion_matrices.png                 # Confusion matrices for all 3 models
│   └── v7_model_comparison.png                   # Accuracy, F1-Score & training time comparison
│
├── .gitignore
└── README.md
```

---

## 📓 What's Inside the Notebook

The notebook (`notebooks/mental_health_text_classification.ipynb`) walks through the full CRISP-DM pipeline in clearly labelled sections:

1. **Data Loading** — Upload and extract the Kaggle dataset from a ZIP file
2. **Exploratory Data Analysis (EDA)** — 5 visualisations: class distribution, pie chart, text length histograms, average length by category, and a word cloud
3. **NLP Preprocessing Pipeline** (6 steps):
   - Drop irrelevant columns
   - Remove null/empty records
   - Lowercase conversion
   - URL, punctuation & special character removal (regex)
   - Tokenization (NLTK `word_tokenize`)
   - Stop word removal + lemmatization (NLTK `WordNetLemmatizer`)
4. **Feature Extraction** — TF-IDF with unigrams + bigrams, `max_features=10,000`, stratified 80/20 train-test split
5. **Model Training** — Multinomial Naive Bayes, Logistic Regression, LinearSVC — all with `class_weight='balanced'`
6. **Evaluation** — Accuracy, weighted F1-Score, training time, full classification reports, and side-by-side confusion matrices

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Environment | Google Colab |
| Data Manipulation | pandas, NumPy |
| NLP | NLTK (tokenization, stopwords, lemmatization) |
| Feature Extraction | scikit-learn `TfidfVectorizer` |
| ML Models | scikit-learn (`MultinomialNB`, `LogisticRegression`, `LinearSVC`) |
| Evaluation | scikit-learn (`classification_report`, `confusion_matrix`, `f1_score`) |
| Visualisation | matplotlib, seaborn, wordcloud |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Download the dataset from Kaggle:  
   [Sentiment Analysis for Mental Health — Suchintika Sarkar](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
3. Upload `Combined Data.csv.zip` when prompted by the first cell
4. Run all cells in order (`Runtime → Run all`)

### Option 2 — Local (Jupyter)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mental-health-nlp-classifier.git
cd mental-health-nlp-classifier

# 2. Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud jupyter

# 3. Download NLTK data (run once)
python -c "import nltk; nltk.download(['punkt','punkt_tab','stopwords','wordnet'])"

# 4. Place 'Combined Data.csv' inside a folder named 'mental_health/' 
#    (skip the Colab upload cell)

# 5. Launch notebook
jupyter notebook notebooks/mental_health_text_classification.ipynb
```

> **Note:** The dataset is not included in this repository. Download it from Kaggle using the link above.

---

## 📁 Dataset

- **Source:** [Kaggle — Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) (Sarkar, 2023)
- **Original size:** 53,043 Reddit posts
- **After cleaning:** 52,681 posts (362 null/empty records removed)
- **Classes:** 7 (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder)
- **Class imbalance:** Normal (30.8%) and Depression (29.0%) dominate; Personality Disorder is the smallest class (2.3%)

---

## 🔬 Methodology

This project follows the **CRISP-DM** (Cross Industry Standard Process for Data Mining) framework — chosen for its iterative, problem-first approach that allows revisiting earlier stages when issues arise (e.g., discovering class imbalance during EDA informed the use of `class_weight='balanced'` during modelling).

---

## ⚠️ Limitations & Future Work

- **23.7% misclassification rate** — not production-ready for clinical or safety-critical use
- Depression and Suicidal posts are linguistically overlapping; all models confused them consistently
- Dataset is English-only; excludes non-English speakers
- Future work: fine-tuned transformer models (**BERT**, **MentalBERT**), multilingual support, and retraining pipelines to address concept drift

---

## 📚 References

- Inamdar et al. (2023). Machine learning driven mental stress detection on Reddit posts using NLP. *HumanCentric Intelligent Systems*
- Muñoz et al. (2022). Comparative analysis of embedding techniques for mental health classification. *Healthcare Technology Letters*
- Ji et al. (2021). MentalBERT: Publicly available pretrained language models for mental healthcare. *arXiv:2110.15621*
- Zhang et al. (2022). NLP applied to mental illness detection: A narrative review. *NPJ Digital Medicine*
- Sarkar, S. (2023). Sentiment Analysis for Mental Health [Dataset]. Kaggle.

---

## 👩‍💻 Author

**Suruthi Kattampalayam Sivasankar**  
BCSI — New INTI International University  
IBM3201 Data Mining & Predictive Analytics · Jan 2026
