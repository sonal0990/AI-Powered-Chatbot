import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

# --- Global variables for the model ---
lemmatizer = WordNetLemmatizer()
vectorizer = None
classifier = None
tags = []
all_patterns = []
responses = {}

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return " ".join(lemmas)

def initialize_nlp_model():
    global vectorizer, classifier, tags, all_patterns, responses

    tags = []
    all_patterns = []
    responses = {}

    with open('data/intents.json', 'r', encoding='utf-8') as file:
        intents_data = json.load(file)

    y_labels = []

    for intent in intents_data['intents']:
        tag = intent['tag']
        tags.append(tag)
        responses[tag] = intent['responses']
        for pattern in intent['patterns']:
            all_patterns.append(preprocess_text(pattern))
            y_labels.append(tag)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(all_patterns)

    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train, y_labels)

    print("NLP Model initialized: TF-IDF vectorizer and Logistic Regression classifier trained.")


def get_chatbot_response(user_message, session_id=None):
    if not vectorizer or not classifier:
        initialize_nlp_model()

    processed_user_message = preprocess_text(user_message)
    user_message_vector = vectorizer.transform([processed_user_message])

    predicted_tag = classifier.predict(user_message_vector)[0]

    probabilities = classifier.predict_proba(user_message_vector)[0]
    confidence = probabilities[classifier.classes_ == predicted_tag][0]

    # --- DEBUGGING PRINTS ---
    print(f"\n--- Chatbot Debug Info ---")
    print(f"User Message: '{user_message}'")
    print(f"Processed Message: '{processed_user_message}'")
    print(f"Predicted Tag: '{predicted_tag}'")
    print(f"Confidence for '{predicted_tag}': {confidence:.4f}")
    print(f"--------------------------\n")

    CONFIDENCE_THRESHOLD = 0.1 # Try lowering this to 0.4 or 0.3 if still struggling.

    if confidence > CONFIDENCE_THRESHOLD:
        return random.choice(responses[predicted_tag])
    else:
        return "I'm sorry, I don't quite understand that. Could you please rephrase or ask something else?"