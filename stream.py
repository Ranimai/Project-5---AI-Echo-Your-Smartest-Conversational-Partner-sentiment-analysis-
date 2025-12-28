import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


with open('confusion_matrix.npy', 'rb') as f:
    cm = np.load(f)
    
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    

st.markdown("""<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Echo: Sentiment Analysis Dashboard </h1>""",unsafe_allow_html=True)
st.divider()

st.subheader("📄**Customer Experience and Business Analytics**")


df = pd.read_csv("cleaned_chatgpt_reviews.csv")
st.write(df.head(10))


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


st.header("🧩 **Confusion Matrix**")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
st.pyplot(fig)


st.header("🔮 **Predict Sentiment**")
user_input = st.text_area("✍️ Enter a review: ")
st.caption("ℹ️ For better accuracy, please enter a complete sentence or review.")

if st.button("🔍**Predict**"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    emoji = {
    "Positive": "😊",
    "Neutral": "😐",
    "Negative": "😞"
    }
    st.success(f"**Predicted Sentiment** :  {prediction} {emoji[prediction]}")

st.divider()
st.markdown("""<h4 style='text-align: center; color: gray;'>Developed by Rani S. | NLP Enthusiast | © 2025</h4>""",unsafe_allow_html=True)

