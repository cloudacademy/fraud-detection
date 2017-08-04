from flask import Flask
from flask import request
import json

from sklearn.externals import joblib

from settings import model_dump_filename

app = Flask(__name__)

engine = joblib.load(model_dump_filename)


@app.route(u"/fraud_score", methods=[u"POST"])
def fraud_score():
    input_data = request.get_json()
    if u"features" not in input_data:
        return json.dumps({u"error": u"No features found in input"}), 400
    if not input_data[u"features"] or not isinstance(input_data[u"features"], list):
        return json.dumps({u"error": u"No feature values available"}), 400
    if isinstance(input_data[u"features"][0], list):
        results = engine.predict_proba(input_data[u"features"]).tolist()
    else:
        results = engine.predict_proba([input_data[u"features"]]).tolist()
    return json.dumps({u"scores": [result[1] for result in results]}), 200
