import os

dir_path = os.path.dirname(os.path.realpath(__file__))

training_data_filename = os.path.join(dir_path, u"data", u"creditcard.csv")
model_dump_filename = os.path.join(dir_path, u"models", u"fraud_detection.pickle")
