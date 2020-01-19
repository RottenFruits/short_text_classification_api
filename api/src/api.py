from flask import Flask, render_template
from flask_cors import CORS
from sklearn.externals import joblib
import flask
import numpy as np
from text_analyzer import TextAnalyzer

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
clf = None
vectorizer = None

def load_model():
    global clf
    global vectorizer
    clf = joblib.load("./model/clf.pkl")
    vectorizer = joblib.load("./model/vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.get_json().get("text"):
        text = flask.request.get_json().get("text")[0]
        ta = TextAnalyzer()
        ma = ta.morphological_analize(text)
        text = " ".join(ma[0])
        print(text)
        feature = vectorizer.transform([text]).toarray()
        response["prediction"] = clf.predict(feature).tolist()
        print(response)

        response["success"] = True
    return flask.jsonify(response)


    
if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=5000)