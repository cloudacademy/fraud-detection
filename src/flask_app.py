from __future__ import unicode_literals
import json
from flask import Flask, request
from sklearn.externals import joblib
from settings import MODEL_FILENAME

app = Flask("Fraud Detection")

# load model at startup time
app.model = joblib.load(MODEL_FILENAME)

@app.route(u"/predict", methods=[u"POST"])
def predict_fraud():
    input_data = request.get_json()
    if u"features" not in input_data:
        return json.dumps({u"error": u"No features found in input"}), 400
    if not input_data[u"features"] or not isinstance(input_data[u"features"], list):
        return json.dumps({u"error": u"No feature values available"}), 400
    if isinstance(input_data[u"features"][0], list):
        results = app.model.predict_proba(input_data[u"features"]).tolist()
    else:
        results = app.model.predict_proba([input_data[u"features"]]).tolist()
    return json.dumps({u"scores": [result[1] for result in results]}), 200
