# fraud-detection
Fraud Detection model based on anonymized credit card transactions


## Getting started
In order to set up a microservice exposing a fraud detection POST endpoint, follow these steps:

1. get the code from the repository
```
git clone https://github.com/cloudacademy/fraud-detection.git 
```
2. [download the dataset](https://clouda-datasets.s3.amazonaws.com/creditcard.csv.zip) that will be used to train a transaction classifier. Unzip it and put the content (creditcard.csv) under folder data

3. create a virtual environment (named e.g. fraud-detection), activate it and retrieve all needed python packages
```
pip install -r requirements-dev.txt
```
In case you do not needed to launch tests associated to this repo you only need to
```
pip install -r requirements.txt
```
instead

4. launch a training for the fraud detection model
```
python src/train.py 
```
from the repo root. This will show information about the advancement of training, the parameters tried during parameter optimization and the quality metrics achieved for different cases. This step should end with a model.pickle file under folder models.

5. launch the Flask app
```
export FLASK_APP=src/flask_app.py
flask run
```
After this, a POST endpoint is exposed at http://127.0.0.1:5000/. You can send an application/json body of the form
```
{
    "features": [
    	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    	[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ]
}
```
i.e. the value of key "features" is a list of 30-floats-long lists, representing the values associated to the transaction.
You will be returned a JSON
```
{
    "scores": [
        0.0323602000089039, 
        0.00037634905230425804
    ]
}
```
i.e. one fraud probability per transaction list submitted

6. if the requirements-dev were installed, you can launch tests for the microservice, via
```
nosetests
```
from the repo root