
# Imports
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Read the data
data_frame = pd.read_csv("AI_Human.csv")

# Allow python access to data attributes
data_frame.shape
data_frame.head()

# Get the scores from data frame
# .generated is the score associated with likeliness to be AI generated
scores = data_frame.generated
scores.head()

# Splitting dataset into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(
    data_frame['text'], scores, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
# Removal of 'filler' words in data for rapidity
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# TFIDFVectorizer essentially analyzes the
# dataframe to compare occurring words to how
# often they occur in the fram to how common they
# are usually.The vectorizor generates a matrix of TF-IDFs
# to visualize keywords for LLM training.


# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Save the vectorizer and classifier to disk
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(pac, "model_pac.pkl")

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

# Build confusion matrix


def matrix():
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Function to test live cases


def check_text(sample_text):
    vec = tfidf_vectorizer.transform([sample_text])
    pred = pac.predict(vec)[0]
    return "AI-Generated" if pred == 1 else "Human-Written"


# print(check_text(input("Enter passage: ")))
