import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Movie Review Analyzer", layout="centered")
st.title("🍿 Movie Review Sentiment Classifier")

uploaded_file = st.file_uploader("Upload Movie Review Dataset (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Convert labels
    data["sentiment"] = data["sentiment"].replace({"positive": 1, "negative": 0})

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    X_train, X_test, y_train, y_test = train_test_split(
        data["review"], data["sentiment"], test_size=0.25, random_state=1
    )

    # Changed vectorizer (CountVectorizer instead of TF-IDF)
    vectorizer = CountVectorizer(stop_words="english", max_features=4000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    predictions = classifier.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    st.success(f"✅ Model Accuracy: {round(accuracy*100,2)}%")

    # Custom prediction section
    st.subheader("✍️ Test Your Own Review")
    user_review = st.text_area("Enter a movie review")

    if st.button("Analyze Review"):
        if user_review.strip() == "":
            st.warning("Please enter a review!")
        else:
            review_vec = vectorizer.transform([user_review])
            result = classifier.predict(review_vec)[0]

            if result == 1:
                st.success("🌟 Positive Review Detected!")
            else:
                st.error("😞 Negative Review Detected!")
