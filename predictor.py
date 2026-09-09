import joblib

tVectorizer = joblib.load("tfidf_vectorizer.pkl")
pacModel = joblib.load("model_pac.pkl")


# Function to test live cases
def check_text(sample_text):
    vec = tVectorizer.transform([sample_text])
    pred = pacModel.predict(vec)[0]
    return "AI-Generated" if pred == 1 else "Human-Written"


print(check_text(input("Paste passage for verification before submission: ")))
