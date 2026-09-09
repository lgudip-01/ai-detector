import json
import os
from http.server import BaseHTTPRequestHandler
import joblib

# Load the saved model and vectorizer
# Ensure these .pkl files are in the same root directory as your repository
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "model_pac.pkl")
vectorizer_path = os.path.join(current_dir, "..", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        try:
            body = json.loads(post_body.decode("utf-8"))
            user_text = body.get("text", "").strip()

            if not user_text:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(
                    {"error": "No text provided"}).encode("utf-8"))
                return

            # Transform input and predict
            vectorized_text = vectorizer.transform([user_text])
            prediction = model.predict(vectorized_text)[0]

            # Map the prediction flag to readable text
            # Assuming 1 = AI-Generated, 0 = Human/Genuine
            label = "AI-Generated" if int(
                prediction) == 1 else "Genuine / Human"

            response_data = {
                "label": label,
                "prediction": int(prediction)
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
