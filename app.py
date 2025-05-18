import warnings
warnings.filterwarnings("ignore")

import os
import csv
import re
import torch
# Prevent Streamlit’s watcher from crashing on torch.classes
torch.classes.__path__ = []

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import plotly.graph_objects as go

import networkx as nx
from pyvis.network import Network
from collections import Counter

import streamlit.components.v1 as components

# ── DRAW PYVIS GRAPH ───────────────────────────────────────────────────────
def draw_pyvis_graph(G, height_px: int = 600):
    """
    Embed a NetworkX DiGraph G as an interactive PyVis force-directed network.
    """
    net = Network(
        height=f"{height_px}px",
        width="100%",
        notebook=False,
        directed=True
    )
    net.from_nx(G)
    net.force_atlas_2based()  # nicer layout
    html_str = net.generate_html()
    components.html(html_str, height=height_px, scrolling=True)


# ── MODEL LOADING ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("logreg_model.pkl", "rb") as f:
        logreg_model = pickle.load(f)
    with open("logreg_vectorizer.pkl", "rb") as f:
        logreg_vectorizer = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm_pipeline = pickle.load(f)
    svm_clf = svm_pipeline.steps[-1][1]
    with open("svm_vectorizer.pkl", "rb") as f:
        svm_vectorizer = pickle.load(f)
    bert_tokenizer = BertTokenizer.from_pretrained("bert_model")
    bert_model = BertForSequenceClassification.from_pretrained("bert_model")
    bert_model.eval()
    return logreg_model, logreg_vectorizer, svm_vectorizer, svm_clf, bert_model, bert_tokenizer

logreg_model, logreg_vectorizer, svm_vectorizer, svm_clf, bert_model, bert_tokenizer = load_models()

# ── PREDICTION HELPER ─────────────────────────────────────────────────────
def predict(text: str, model_choice: str):
    if model_choice == "Logistic Regression":
        X = logreg_vectorizer.transform([text])
        pred = logreg_model.predict(X)[0]
        probs = logreg_model.predict_proba(X)[0]
    elif model_choice == "SVM":
        X = svm_vectorizer.transform([text])
        pred = svm_clf.predict(X)[0]
        try:
            probs = svm_clf.predict_proba(X)[0]
        except AttributeError:
            p = expit(svm_clf.decision_function(X)[0])
            probs = [1 - p, p]
    else:  # BERT
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs_tensor = torch.nn.functional.softmax(logits, dim=1)[0]
        probs = probs_tensor.cpu().numpy()
        pred = int(probs_tensor.argmax())
    label = "Positive" if pred == 1 else "Negative"
    confidence = float(max(probs))
    return label, confidence, {"Negative": float(probs[0]), "Positive": float(probs[1])}

# ── LOGGING SETUP ──────────────────────────────────────────────────────────
LOG_PATH = "tweet_logs.csv"
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp", "model", "text", "prediction", "confidence"]
        )

def log_input(model: str, text: str, label: str, conf: float):
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, model, text, label, f"{conf:.4f}"])


# ── APP LAYOUT ─────────────────────────────────────────────────────────────
st.title("🚀 Tweet Analysis Suite")

tabs = st.tabs(["Sentiment Classifier", "Knowledge Graph"])

# ── TAB 1: SENTIMENT CLASSIFIER ────────────────────────────────────────────
with tabs[0]:
    st.sidebar.title("Settings")
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    if st.sidebar.button("🌙 Toggle Dark/Light", key="toggle_theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode

    if st.session_state.dark_mode:
        st.markdown(
            """
            <style>
              .css-1d391kg, .css-12oz5g7 { background-color: #1e1e1e !important; color: #f1f1f1; }
              .stTextInput, .stSelectbox, .stButton button { background-color: #2e2e2e !important; color: white; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    model_choice = st.sidebar.selectbox(
        "Model", ["BERT", "Logistic Regression", "SVM"], key="model_choice"
    )
    csv_file = st.sidebar.file_uploader(
        "Upload CSV (no header, 6 cols)", type="csv", key="csv_uploader"
    )
    run_button = st.sidebar.button("Run predictions", key="run_batch")

    st.subheader("📝 Tweet Sentiment Classifier")

    # Batch predictions
    if csv_file and run_button:
        try:
            df = pd.read_csv(
                csv_file,
                header=None,
                names=["label", "id", "date", "flag", "user", "text"],
                encoding="latin1",
            )
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        results = []
        for _, row in df.iterrows():
            text = str(row["text"])
            true_label = row["label"]
            pred_label, conf, probs = predict(text, model_choice)
            log_input(model_choice, text, pred_label, conf)
            results.append({
                "text": text,
                "true_label": true_label,
                "prediction": pred_label,
                "confidence": conf,
                "Neg_prob": probs["Negative"],
                "Pos_prob": probs["Positive"],
            })

        out_df = pd.DataFrame(results)
        st.success("Predictions complete!")
        st.dataframe(out_df[["text", "true_label", "prediction", "confidence"]])

        st.download_button(
            "Download results as CSV",
            data=out_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            key="download_results"
        )

        # Metrics
        out_df["true_binary"] = out_df["true_label"].apply(lambda x: 1 if int(x) == 4 else 0)
        out_df["pred_binary"] = (out_df["prediction"] == "Positive").astype(int)
        y_true = out_df["true_binary"]
        y_pred = out_df["pred_binary"]
        y_score = out_df["Pos_prob"]

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report_dict = classification_report(
            y_true, y_pred,
            target_names=["Negative", "Positive"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df["support"] = report_df["support"].astype(int)

        st.markdown("---")
        st.subheader("Overall Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [f"{acc:.2%}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]
        })
        st.table(metrics_df)

        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Pred Negative", "Pred Positive"]
        )
        st.table(cm_df)

        st.subheader("Classification Report")
        st.table(report_df)

        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    # Single prediction
    tweet = st.text_area("Your tweet here:", key="single_tweet")
    if st.button("Predict", key="single_predict"):
        if not tweet.strip():
            st.warning("Please enter some text.")
        else:
            lbl, cf, probs = predict(tweet, model_choice)
            if lbl == "Positive":
                st.markdown("### ✅ Positive sentiment!")
                st.success("This tweet reads as positive.")
            else:
                st.markdown("### ❌ Negative sentiment.")
                st.error("This tweet reads as negative.")

            prob_df = pd.DataFrame.from_dict(
                probs, orient="index", columns=["probability"]
            )
            st.bar_chart(prob_df)
            st.caption(f"Confidence: **{cf:.2%}**")

            fig3 = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=cf * 100,
                    title={"text": "Confidence (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"thickness": 0.3},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 100], "color": "lightgreen"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig3, use_container_width=True)
            log_input(model_choice, tweet, lbl, cf)

    st.markdown(
        "<hr><center>🚀 Built with ❤️ using Streamlit</center>",
        unsafe_allow_html=True
    )

# ── TAB 2: INTERACTIVE USER & KEYWORD GRAPH ────────────────────────────────
with tabs[1]:
    st.subheader("🔍 Interactive Mention/Keyword Graph (PyVis)")
    DATA_PATH = "/Users/yashwanthreddy/Desktop/ML End Sem Project/Dataset.csv"

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    # Load full dataset once
    df_full = pd.read_csv(
        DATA_PATH,
        header=None,
        names=["label", "id", "date", "flag", "user", "text"],
        encoding="latin1",
    )

    # Shared inputs
    username = st.text_input("Filter by username (no @)", key="kg_user").strip().lower()
    keyword  = st.text_input("Filter by keyword", key="kg_keyword").strip().lower()
    max_rows = st.slider(
        "Max rows to consider",
        1000,
        min(200000, len(df_full)),
        20000,
        1000,
        key="kg_sample"
    )

    if not username and not keyword:
        st.info("Enter a username or keyword above to build the graph.")
    else:
        # Sample for performance
        df_sample = df_full.sample(min(len(df_full), max_rows), random_state=1)
        edges = Counter()

        # Build edge counts
        if username:
            title = f"@{username} → mentions"
            mask_user = df_sample["user"].str.lower() == username
            for _, row in df_sample[mask_user].iterrows():
                for tgt in re.findall(r"@(\w+)", str(row["text"])):
                    edges[(username, tgt.lower())] += 1
        else:
            title = f"Keyword '{keyword}' → users"
            mask_key = df_sample["text"].str.lower().str.contains(keyword)
            for _, row in df_sample[mask_key].iterrows():
                edges[(keyword, row["user"].lower())] += 1

        top_edges = edges.most_common(50)
        if not top_edges:
            st.warning("No edges to display.")
        else:
            # Construct NetworkX graph
            G = nx.DiGraph()
            for (u, v), w in top_edges:
                G.add_edge(u, v, weight=w)

            st.markdown(f"**Graph:** {title}")
            # Draw interactive PyVis network
            draw_pyvis_graph(G, height_px=600)

            # Show the exact tweets that went into this graph
            if username:
                matched = df_sample[mask_user][["user", "text", "date"]].copy()
            else:
                matched = df_sample[mask_key][["user", "text", "date"]].copy()

            # Annotate each tweet with sentiment
            matched["Sentiment"] = matched["text"].apply(lambda t: predict(t, model_choice)[0])
            matched["Confidence"] = matched["text"].apply(lambda t: predict(t, model_choice)[1])

            st.markdown("### 📜 Matching Tweets with Sentiment")
            if matched.empty:
                st.info("No tweets matched your filter.")
            else:
                with st.expander(f"Show {len(matched)} raw tweets"):
                    st.dataframe(
                        matched.rename(columns={
                            "user": "User",
                            "text": "Tweet Text",
                            "date": "Date"
                        }),
                        height=300
                    )
