from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def load_sms(uploaded_file=None):
    """Supports Kaggle spam.csv, standard SMS CSV, or the public SMS TSV mirror."""
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, sep='\t', header=None, names=['label', 'message'])
        else:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        local = Path(__file__).with_name('spam.csv')
        local_tsv = Path(__file__).with_name('sms.tsv')
        if local.exists():
            df = pd.read_csv(local, encoding='latin-1')
        elif local_tsv.exists():
            df = pd.read_csv(local_tsv, sep='\t', header=None, names=['label', 'message'])
        else:
            df = pd.read_csv(DATA_URL, sep='\t', header=None, names=['label','message'])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'v1' in df and 'v2' in df:
        df = df.rename(columns={'v1':'label', 'v2':'message'})
    if 'label' not in df or 'message' not in df:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]:'label', df.columns[1]:'message'})
        else:
            raise ValueError('CSV needs label and message columns (or v1 and v2).')
    df = df[['label','message']].dropna().copy()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['message'] = df['message'].astype(str)
    df = df[df['label'].isin(['spam','ham'])]
    if df.empty or df['label'].nunique() < 2:
        raise ValueError('Dataset must contain both ham and spam labels.')
    return df.reset_index(drop=True)


def apply_style():
    st.markdown("""
    <style>
      .block-container {max-width: 1100px; padding-top: 2.6rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.025em;}
      h1 {font-size: 2.15rem !important; font-weight: 700 !important; margin-bottom: 0.2rem !important;}
      h3 {font-size: 1.10rem !important; font-weight: 650 !important;}
      [data-testid="stMetric"] {background: rgba(128, 128, 128, .045); border: 1px solid rgba(128,128,128,.17); border-radius: 9px; padding: 13px 16px;}
      [data-testid="stMetricLabel"] {font-size: .85rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 650;}
      div[data-testid="stButton"] button[kind="primary"] {border-radius: 7px; font-weight: 600;}
      div[data-testid="stDataFrame"] {border-radius: 8px; overflow: hidden;}
      .eyebrow {font-size: .75rem; color: #64748b; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .65rem;}
      .subtle {font-size: .89rem; color: #64748b; margin-bottom: 1.5rem;}
      .footer-rule {border-top: 1px solid rgba(128,128,128,.2); margin-top: 1.9rem; padding-top: .8rem; color:#64748b; font-size:.8rem;}
    </style>
    """, unsafe_allow_html=True)

def intro(project, title, subtitle):
    st.markdown(f'<div class="eyebrow">{project}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)


st.set_page_config(page_title="Spam classifier | ML", page_icon="◧", layout="wide")
apply_style()
intro("MACHINE LEARNING / MINI PROJECT 02", "SMS Spam Classifier", "A text classification model built with TF-IDF and Multinomial Naive Bayes.")
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload a CSV or TSV", type=["csv", "tsv"])
    st.caption("Optional. Uses the public SMS dataset when no file is uploaded.")
    st.divider()
    st.markdown("**Model configuration**")
    st.caption("Train / test: 80% / 20%")
    st.caption("Feature extraction: TF-IDF")
    st.caption("Algorithm: Multinomial Naive Bayes")
    st.caption("Random state: 42")
try:
    df = load_sms(upload)
    X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=.2, random_state=42, stratify=df["label"])
    model = Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("nb", MultinomialNB())])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    a,b,c,d = st.columns(4)
    a.metric("Accuracy", f"{accuracy_score(y_test, predictions):.1%}")
    b.metric("Spam precision", f"{precision_score(y_test, predictions, pos_label='spam', zero_division=0):.1%}")
    c.metric("Spam recall", f"{recall_score(y_test, predictions, pos_label='spam', zero_division=0):.1%}")
    d.metric("Test messages", f"{len(X_test):,}")
    st.divider()
    tab_predict, tab_evaluate = st.tabs(["Test a message", "Model evaluation"])
    with tab_predict:
        st.subheader("Classify an SMS")
        message = st.text_area("Message", placeholder="Enter a message to check...", height=115)
        if st.button("Check message", type="primary"):
            if message.strip():
                prediction = model.predict([message])[0]
                spam_index = list(model.classes_).index("spam")
                score = model.predict_proba([message])[0][spam_index]
                if prediction == "spam":
                    st.error(f"Result: SPAM · Estimated spam probability: {score:.1%}")
                else:
                    st.success(f"Result: NOT SPAM · Estimated spam probability: {score:.1%}")
            else:
                st.warning("Enter a message before checking it.")
        st.markdown("#### Three sample predictions")
        examples = [
            "Hi, are we still meeting at 6 this evening?",
            "Congratulations! You've won a free cash prize. Claim today!",
            "URGENT: Click this link to collect your guaranteed reward."
        ]
        st.dataframe(pd.DataFrame({"Message": examples, "Prediction": model.predict(examples)}), hide_index=True, use_container_width=True)
    with tab_evaluate:
        left,right = st.columns([1,1], gap="large")
        with left:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
            fig, ax = plt.subplots(figsize=(5,4))
            ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues", colorbar=False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.subheader("Classification report")
            report = classification_report(y_test, predictions, labels=["ham", "spam"], output_dict=True, zero_division=0)
            rows = pd.DataFrame(report).T.loc[["ham", "spam"], ["precision", "recall", "f1-score", "support"]]
            st.dataframe(rows.style.format({"precision":"{:.3f}", "recall":"{:.3f}", "f1-score":"{:.3f}", "support":"{:.0f}"}), use_container_width=True)
            st.caption(f"Training samples: {len(X_train):,} · Test samples: {len(X_test):,}")
    st.markdown('<div class="footer-rule">Educational model · Train/test split is stratified and reproducible.</div>', unsafe_allow_html=True)
except Exception as exc:
    st.error(f"Unable to train the classifier: {exc}")
    st.caption("Upload spam.csv from the SMS Spam Collection dataset, or save it beside app.py.")
