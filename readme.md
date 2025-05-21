# Network-Based Social Media Sentiment Analysis on Twitter

**Tweet Analysis Suite** is an end-to-end sentiment analysis pipeline and interactive dashboard for Twitter data, designed and developed collaboratively by a six-member team: **\[Your Name]**, **\[Teammate A]**, **\[Teammate B]**, **\[Teammate C]**, **\[Teammate D]**, and **\[Teammate E]**.

## 🔍 Project Overview

* **Objective:** Clean, model, and visualize sentiment from Twitter at scale, combining classical ML and transformer models with network-based insights.
* **Dataset:** 1.6M labeled tweets from Sentiment140.
* **Key Technologies:** Python, Streamlit, scikit-learn, Hugging Face Transformers (BERT), NetworkX, PyVis.

## 🚀 Features

1. **Exploratory Data Analysis**

   * Class balance verification, duplicate removal, missing-value checks.
   * Tweet-length distribution, hashtag frequency, word clouds by sentiment.
2. **Text Preprocessing**

   * Regex-based cleaning: lowercasing, URL/mention/hashtag stripping, whitespace normalization.
3. **Modeling Pipeline**

   * **Classical Models:** TF-IDF vectorization + Logistic Regression & SVM.
   * **Deep Learning:** Fine-tuned BERT with 128-token limit.
   * Performance metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC.
4. **Interactive Streamlit App**

   * Dark/Light mode toggle, model selector (BERT/SVM/LR).
   * Single-tweet inference with probability bar charts and confidence gauges.
   * Batch CSV predictions: inline metrics, downloadable results.
5. **Network Visualization**

   * Username and keyword filtering to build directed mention/keyword graphs.
   * Interactive force-directed layouts with drill-down raw-tweet views.
6. **Logging & Monitoring**

   * Audit trail of all predictions in `tweet_logs.csv` (UTC timestamp, model, text, label, confidence).

## 💻 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/tweet-analysis-suite.git
   cd tweet-analysis-suite
   ```
2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\\Scripts\\activate   # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Download and place data**

   * Download `training.1600000.processed.noemoticon.csv` from Sentiment140.
   * Place the file at `data/` or update `DATA_PATH` in `app.py`.

## ▶️ Usage

```bash
streamlit run app.py
```

* Navigate to **Sentiment Classifier** for single or batch predictions.
* Switch to **Knowledge Graph** for network-based visualizations.

## 📂 Repository Structure

```
├── app.py               # Main Streamlit application
├── preprocessing.py     # Text cleaning and EDA scripts
├── train_models.py      # Scripts to train LR, SVM, and fine-tune BERT
├── models/              # Serialized model and tokenizer files
├── data/                # Raw and processed datasets
├── requirements.txt     # Python dependencies
├── tweet_logs.csv       # Prediction audit logs (auto-generated)
└── README.md            # Project documentation
```


