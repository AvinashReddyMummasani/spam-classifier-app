import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

ps = PorterStemmer()

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Text preprocessing function
def transfrom_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english"):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if input_sms:
    # 1. Preprocess
    transformed_sms = transfrom_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Output
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
