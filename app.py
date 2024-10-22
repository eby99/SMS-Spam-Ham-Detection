import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

# Define the custom NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Append NLTK data path
nltk.data.path.append(nltk_data_path)

# Download necessary resources to the custom directory
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Ensure punkt_tab is downloaded (though it's not usually necessary)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)

# Initialize the Porter Stemmer
port_stemmer = PorterStemmer()

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def clean_text(text):
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem the tokens
    tokens = [word.lower() for word in tokens if word.lower() not in set(stopwords.words('english'))]
    tokens = list(map(lambda x: port_stemmer.stem(x), tokens))
    
    # Join the tokens back together
    return " ".join(tokens)

# Streamlit app title
st.title('SMS Spam Classifier')

# User input for SMS message
input_sms = st.text_input("Enter the Message")

if st.button('Predict'):
    if input_sms == "":
        st.header('Please Enter Your Message !!!')
    else:
        # Preprocess the input message
        transform_text = clean_text(input_sms)
        
        # Vectorize the cleaned text
        vector_input = tfidf.transform([transform_text])
        
        # Make a prediction
        result = model.predict(vector_input)
        
        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
