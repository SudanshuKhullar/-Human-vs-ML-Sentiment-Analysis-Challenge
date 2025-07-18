
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
import pickle
import matplotlib.pyplot as plt # Import matplotlib for the chart

# --- Custom CSS for Styling ---
custom_css = """
<style>
body {
    background-color: #000; /* Black background */
    color: #fff; /* White text for contrast */
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}
.stApp {
    background-color: #000; /* Ensure the main app area also has the background */
    color: #fff; /* White text for contrast */
}
.stButton>button {
    background-color: #4CAF50; /* Green button */
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border: none;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049; /* Darker green on hover */
}
.stRadio > label > div {
    font-size: 16px;
    margin-right: 15px;
}
.stAlert {
    border-radius: 5px;
    padding: 10px;
}
.stAlert.success {
    background-color: #d4edda;
    color: #155724;
    border-color: #c3e6cb;
}
.stAlert.error {
    background-color: #f8d7da;
    color: #721c24;
    border-color: #f5c6cb;
}
.stMetric > div > div:first-child {
    font-size: 18px;
    color: #bbb; /* Lighter grey for metric labels */
}
.stMetric > div > div:last-child {
    font-size: 30px;
    font-weight: bold;
    color: #fff; /* White for metric values */
}
h1, h2, h3, h4, h5, h6 {
    color: #00b0f0; /* A brighter color for headings on black background */
}
/* Style for the info box displaying the review */
.stAlert.info {
    background-color: #333; /* Dark grey for review box */
    color: #eee; /* Light grey text */
    border-color: #555;
    font-style: italic;
}
</style>
"""

# --- Helper Functions ---

def preprocess_text(text):
    """Cleans the input text."""
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# --- Load Data and Train Model (replace with loading saved model in a real app) ---
# In a real deployment, you would load a pre-trained model and vectorizer
# to keep the app fast. For demonstration, we'll include training here.

@st.cache_resource
def load_and_train_model():
    """Loads data and trains the model."""
    # Load the dataset (assuming 'archive (1).zip' was uploaded and is accessible)
    # In a deployed app, you would load the data from a more reliable source
    try:
        # Adjust the path if your file is located elsewhere
        df = pd.read_csv('archive (1).zip')
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure 'archive (1).zip' is in the same directory or update the path.")
        return None, None, None

    # Check if the expected columns exist after loading
    if 'review' not in df.columns or 'sentiment' not in df.columns:
         # Assuming the columns might be unnamed and in position 0 and 1
         if df.shape[1] >= 2:
             df.rename(columns={df.columns[0]: 'review', df.columns[1]: 'sentiment'}, inplace=True)
         else:
            st.error("Dataset does not contain 'review' and 'sentiment' columns. Please check your CSV file.")
            return None, None, None


    df['cleaned_review'] = df['review'].apply(preprocess_text)
    # Ensure sentiment column is treated as string before applying lower()
    df['sentiment'] = df['sentiment'].astype(str).str.lower()
    y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)


    # Train TF-IDF Vectorizer (fit on the entire dataset)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    X = tfidf_vectorizer.fit_transform(df['cleaned_review'])

    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, tfidf_vectorizer, df

model, tfidf_vectorizer, df = load_and_train_model()

if model is not None:
    # --- Streamlit App ---

    st.set_page_config(page_title="Human vs ML Sentiment Challenge", layout="wide")

    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("🎬 Human vs ML Sentiment Analysis Challenge")

    st.markdown("""
    Welcome to the challenge! Can your sentiment analysis skills beat our trained machine learning model?

    Read the movie review below and predict if the sentiment is **Positive** or **Negative**.
    Then, submit your prediction to see the ML model's result and the actual sentiment!
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Sidebar ---
    st.sidebar.header("About the Challenge")
    st.sidebar.markdown("""
    This interactive application allows you to challenge a machine learning model in predicting the sentiment of movie reviews from the IMDb dataset.

    *   **Your Goal:** Correctly classify the sentiment (positive or negative) of each review.
    *   **ML Model's Goal:** Predict the sentiment using a trained Logistic Regression model.
    *   **Compare:** See how your accuracy stacks up against the model's!
    """)

    # Use an expander for the history section
    with st.sidebar.expander("Challenge History"):
        if 'human_predictions' in st.session_state and st.session_state.human_predictions:
            for i in range(len(st.session_state.human_predictions)):
                review_num = i + 1
                human_pred = st.session_state.human_predictions[i]
                ml_pred = st.session_state.ml_predictions[i]
                actual = st.session_state.actual_sentiments[i]

                human_correct_icon = "✅" if human_pred.lower() == actual.lower() else "❌"
                ml_correct_icon = "✅" if ml_pred.lower() == actual.lower() else "❌"

                st.markdown(f"**Review {review_num}:** You {human_correct_icon}, ML {ml_correct_icon}")
        else:
            st.info("Make your first prediction to see history here!")

    st.sidebar.subheader("Model Details")
    st.sidebar.markdown("""
    *   **Type:** Logistic Regression
    *   **Trained on:** 50,000 IMDb movie reviews
    *   **Features:** TF-IDF Vectorization (Top 5000 features)
    """)


    # --- Select a review for the challenge ---
    if 'challenge_reviews' not in st.session_state or 'current_review_index' not in st.session_state:
        # Ensure df is not None before sampling
        if df is not None:
            st.session_state.challenge_reviews = df.sample(n=10, random_state=random.randint(0,10000)).to_dict('records') # Select 10 random reviews
            st.session_state.current_review_index = 0
            st.session_state.human_predictions = []
            st.session_state.ml_predictions = []
            st.session_state.actual_sentiments = []
        else:
             st.error("Could not load dataset to start the challenge.")


    if 'challenge_reviews' in st.session_state and len(st.session_state.challenge_reviews) > 0:
        current_review_data = st.session_state.challenge_reviews[st.session_state.current_review_index]
        current_review = current_review_data['review']
        actual_sentiment = current_review_data['sentiment']

        st.subheader(f"Challenge Review {st.session_state.current_review_index + 1} of {len(st.session_state.challenge_reviews)}")
        st.info(current_review) # Use st.info for a distinct look

        # Display current human score (outside sidebar for prominence)
        current_human_correct = sum(1 for hp, actual in zip(st.session_state.human_predictions, st.session_state.actual_sentiments) if hp.lower() == actual.lower())
        st.write(f"Your current score: **{current_human_correct} / {st.session_state.current_review_index}** correct predictions")


        # --- Human Prediction ---
        st.markdown("### Your Prediction")
        human_prediction = st.radio("Choose the sentiment:", ('Positive', 'Negative'), key=f"human_pred_{st.session_state.current_review_index}", horizontal=True)

        # --- Make and show ML Prediction (only after human predicts) ---
        if st.button("Submit Prediction and See ML Result", key=f"submit_btn_{st.session_state.current_review_index}"):
            # Get ML prediction
            cleaned_review = preprocess_text(current_review)
            X_single = tfidf_vectorizer.transform([cleaned_review])
            ml_prediction_numeric = model.predict(X_single)[0]
            ml_prediction = 'Positive' if ml_prediction_numeric == 1 else 'Negative'

            # Store results
            st.session_state.human_predictions.append(human_prediction)
            st.session_state.ml_predictions.append(ml_prediction)
            st.session_state.actual_sentiments.append(actual_sentiment)

            # Display results for the current review in columns
            st.markdown("---")
            st.subheader("Results for this Review")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Your Prediction:**")
                if human_prediction.lower() == actual_sentiment.lower():
                     st.success(human_prediction)
                else:
                     st.error(human_prediction)


            with col2:
                st.write(f"**ML Model Prediction:**")
                if ml_prediction.lower() == actual_sentiment.lower():
                    st.success(ml_prediction)
                else:
                    st.error(ml_prediction)

            with col3:
                 st.write(f"**Actual Sentiment:**")
                 st.write(actual_sentiment.capitalize())


            st.markdown("---")

            # Move to the next review or finish
            if st.session_state.current_review_index < len(st.session_state.challenge_reviews) - 1:
                st.session_state.current_review_index += 1
                # Add a button to go to the next review
                st.button("Next Review", key=f"next_btn_{st.session_state.current_review_index}", on_click=st.rerun)
            else:
                st.subheader("Challenge Complete!")
                st.write("You have finished all the reviews in this challenge.")

                # Calculate and display final scores
                human_correct_count = sum(1 for hp, actual in zip(st.session_state.human_predictions, st.session_state.actual_sentiments) if hp.lower() == actual.lower())
                ml_correct_count = sum(1 for mlp, actual in zip(st.session_state.ml_predictions, st.session_state.actual_sentiments) if mlp.lower() == actual.lower())
                total_reviews = len(st.session_state.challenge_reviews)

                human_accuracy = (human_correct_count / total_reviews) * 100 if total_reviews > 0 else 0
                ml_accuracy = (ml_correct_count / total_reviews) * 100 if total_reviews > 0 else 0

                st.subheader("Final Scores")

                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.metric("Your Accuracy", f"{human_accuracy:.2f}%")
                with score_col2:
                    st.metric("ML Model Accuracy", f"{ml_accuracy:.2f}%")

                # Display accuracy comparison chart (Pie Chart)
                accuracies = [human_accuracy, ml_accuracy]
                labels = ['Your Accuracy', 'ML Model Accuracy']

                # Create pie chart
                fig, ax = plt.subplots(figsize=(6, 6)) # Reduced size
                ax.pie(accuracies, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                ax.set_title('Human vs ML Model Accuracy Comparison')

                st.pyplot(fig)


                # Display comparison table
                # Check if the lengths match before creating the DataFrame
                if len(st.session_state.human_predictions) == len(st.session_state.challenge_reviews):
                    comparison_data = {
                        'Review': [r['review'] for r in st.session_state.challenge_reviews],
                        'Actual Sentiment': st.session_state.actual_sentiments,
                        'Your Prediction': st.session_state.human_predictions,
                        'ML Prediction': st.session_state.ml_predictions,
                        'You Correct': [hp.lower() == actual.lower() for hp, actual in zip(st.session_state.human_predictions, st.session_state.actual_sentiments)],
                        'ML Correct': [mlp.lower() == actual.lower() for mlp, actual in zip(st.session_state.ml_predictions, st.session_state.actual_sentiments)]
                    }
                    comparison_df = pd.DataFrame(comparison_data)
                    st.subheader("Detailed Comparison")
                    st.dataframe(comparison_df)
                else:
                    st.warning("Cannot display detailed comparison until all reviews are completed.")


                # Option to start a new challenge
                st.markdown("---")
                if st.button("Start New Challenge"):
                    # Ensure df is not None before sampling
                    if df is not None:
                        st.session_state.challenge_reviews = df.sample(n=10, random_state=random.randint(0,10000)).to_dict('records')
                        st.session_state.current_review_index = 0
                        st.session_state.human_predictions = []
                        st.session_state.ml_predictions = []
                        st.session_state.actual_sentiments = []
                        st.rerun()
                    else:
                        st.error("Could not load dataset to start a new challenge.")
    elif 'challenge_reviews' in st.session_state and len(st.session_state.challenge_reviews) == 0:
        st.warning("No reviews loaded for the challenge. Please check the dataset.")

    # --- Add a Section for User to Analyze Their Own Text ---
    st.markdown("---")
    st.subheader("Analyze Your Own Review")
    st.write("Want to see what the ML model predicts for a review of your choice? Paste your text below!")

    user_text = st.text_area("Enter your movie review here:", height=150)

    if st.button("Get ML Sentiment Prediction for Your Text"):
        if user_text:
            # Preprocess the user's text
            cleaned_user_text = preprocess_text(user_text)
            X_user_text = tfidf_vectorizer.transform([cleaned_user_text])

            # Get ML prediction
            user_ml_prediction_numeric = model.predict(X_user_text)[0]
            user_ml_prediction = 'Positive' if user_ml_prediction_numeric == 1 else 'Negative'

            st.markdown("---")
            st.subheader("ML Model Prediction for Your Text")
            if user_ml_prediction == 'Positive':
                st.success(f"The ML model predicts: **{user_ml_prediction}**")
            else:
                st.error(f"The ML model predicts: **{user_ml_prediction}**")
        else:
            st.warning("Please enter some text to get a prediction.")
