# -Human-vs-ML-Sentiment-Analysis-Challenge
Read the movie review below and predict if the sentiment is Positive or Negative. Then, submit your prediction to see the ML model's result and the actual sentiment!

This project creates an interactive web application using Streamlit where users can challenge a machine learning model in predicting the sentiment (positive or negative) of movie reviews.

Project Description
The application presents users with movie reviews from the IMDb dataset. Users make their prediction, and then the application reveals the actual sentiment and the prediction made by a trained machine learning model. It also tracks the user's score and provides a final comparison at the end of a challenge session.

Features
Interactive challenge interface.
Sentiment prediction by a trained ML model (Logistic Regression).
Comparison of human vs. ML accuracy.
Challenge history display in the sidebar.
Option to analyze the sentiment of custom review text.
Visually appealing design using custom CSS.
Files in this Repository
game.py: The main Streamlit application script.
requirements.txt: Lists the Python libraries required to run the application.
archive (1).zip: The IMDb dataset used for training the model.

HOW TO RUN LOCALLY:-
1. python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
2.     pip install -r requirements.txt
3.     streamlit run game.py
