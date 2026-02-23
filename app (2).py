import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="IMDB Sentiment", layout="centered")
st.title("🎬 IMDB Sentiment Analysis")

file = st.file_uploader("Upload IMDB CSV", type="csv")
if file:
    df = pd.read_csv(file)
    df["sentiment"] = df["sentiment"].map({"positive":1,"negative":0})

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    st.success(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
