import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('stopwords')

port_stemmer = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in set(stopwords.words('english'))]
    tokens = list(map(lambda x: port_stemmer.stem(x), tokens))
    return " ".join(tokens)

st.title('SMS Spam Classifier')

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):
    if input_sms == "":
        st.header('Please Enter Your Message !!!')
    else:
        transform_text = clean_text(input_sms)
        vector_input = tfidf.transform([transform_text])
        result = model.predict(vector_input)
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")