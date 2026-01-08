import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

    
with open('svm_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)  
    

st.markdown("""<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Echo: Sentiment Analysis Dashboard </h1>""",unsafe_allow_html=True)
st.divider()

st.subheader("📄**Customer Experience and Business Analytics**")


df = pd.read_csv("cleaned_chatgpt_reviews.csv")
st.write(df.head(10))

# st.write("Vectorizer features:", len(vectorizer.get_feature_names_out()))

st.header("⭐ **Rating Distribution**")
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
fig, ax = plt.subplots()
df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax, color=colors)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)


st.header("📊 **Sentiment Distribution**")
colors = ['red', 'green', 'yellow']
fig, ax = plt.subplots()
df['sentiment'].value_counts().plot(kind='bar',color=colors, ax=ax)
st.pyplot(fig)


model_choice = st.selectbox(
    "Select Prediction Model",
    ["SVM", "VADER"]
)

st.header("🔮 **Predict Sentiment**")
user_input = st.text_area("✍️ Enter a review: ")
st.caption("ℹ️ For better accuracy, please enter a complete sentence or review.")

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


if st.button("🔍 Predict"):
    
    # SVM model
    if model_choice == "SVM":
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        pred_num = model.predict(input_vector)[0]
        pred_label = label_map[pred_num]
    
    #  VADER model
    elif model_choice == "VADER":
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(user_input)['compound']

        if score >= 0.05:
            pred_label = "Positive"
        elif score <= -0.05:
            pred_label = "Negative"
        else:
            pred_label = "Neutral"

    # EMOJI mapping
    emoji = {
        "Positive": "😊",
        "Neutral": "😐",
        "Negative": "😞"
    }

    st.success(f"**Predicted Sentiment:** {pred_label} {emoji[pred_label]}")
    

st.divider()

st.markdown("""<h4 style='text-align: center; color: gray;'>Developed by Rani S. | NLP Enthusiast | © 2025</h4>""",unsafe_allow_html=True)

