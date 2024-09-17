import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer

# Load models and vectorizer
cv = pickle.load(open('countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('scalerr.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))
model_xgb = pickle.load(open('model_xgb.pkl', 'rb'))
model_dt = pickle.load(open('model_dt.pkl', 'rb'))

# Set up the Streamlit apps
st.title('Sentiment Analysis Web App')

st.write('Enter your review below and choose a model to predict sentiment.')

# Input from user
review_text = st.text_area("Enter your review here:")

model_choice = st.selectbox(
    'Select the model for prediction:',
    ['RandomForest', 'XGBoost', 'DecisionTree']
)

if st.button('Predict'):
    if review_text:
        # Preprocess the input text
        review = [review_text]
        X = cv.transform(review).toarray()
        X_scl = scaler.transform(X)

        # Choose the model based on user selection
        if model_choice == 'RandomForest':
            prediction = model_rf.predict(X_scl)
        elif model_choice == 'XGBoost':
            prediction = model_xgb.predict(X_scl)
        elif model_choice == 'DecisionTree':
            prediction = model_dt.predict(X_scl)

        # Show prediction
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        st.write(f'The sentiment of the review is: {sentiment}')
    else:
        st.write('Please enter a review text.')
#streamlit run sentiment.py