import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Load your trained model and other necessary objects ---
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    Total_stopwords = stopwords.words('english')
    Total_stopwords.remove('not')
    Total_stopwords.remove('no')
    Total_stopwords.remove('nor')
    Total_stopwords.remove('against')
    Total_stopwords.remove("wasn't")
    Total_stopwords.remove("weren't")
    review = [ps.stem(word) for word in review if not word in set(Total_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
classifier.fit(X, Y)

# --- Streamlit UI ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon=":smile:") 

st.title("Sentiment Analysis")

st.markdown("##  What is this project about?")
st.write("This application leverages the power of machine learning to determine the sentiment expressed within a given text. By analyzing the words and phrases used, it can accurately classify the sentiment as either positive or negative, providing valuable insights for understanding public opinion, customer feedback, and more.")

with st.form(key='sentiment_form'):
    review = st.text_area("Enter your review here (like about the food at a restaurant):", height=150)
    submit_button = st.form_submit_button(label='Analyze')

if submit_button:
    if review:
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        Total_stopwords = stopwords.words('english')
        Total_stopwords.remove('not')
        Total_stopwords.remove('no')
        Total_stopwords.remove('nor')
        Total_stopwords.remove('against')
        Total_stopwords.remove("wasn't")
        Total_stopwords.remove("weren't")
        review = [ps.stem(word) for word in review if not word in
                  set(Total_stopwords)]
        review = ' '.join(review)
        review = cv.transform([review]).toarray()
        prediction = classifier.predict(review)
        if prediction[0] == 1:
            result = "Positive Review 😄" 
            st.success(f"Prediction: {result}") 
        else:
            result = "Negative Review 😔"
            st.error(f"Prediction: {result}")   
    else:
        st.warning("Please enter a review.")

