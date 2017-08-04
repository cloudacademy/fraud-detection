from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import recall_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pandas as pd

from settings import training_data_filename, model_dump_filename

df = pd.read_csv(training_data_filename, header=0)
X = df[df.columns[:-1]].as_matrix()
y = df[df.columns[-1]].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    # TODO some smart preproc can be added here
    (u"clf", LogisticRegression(class_weight=u"balanced")),  # possibly relevant setting
])
parameters = {u"clf__C": (1e-4, 1e-2, 1e0, 1e2, 1e4)}
grid_search = GridSearchCV(pipeline, parameters, scoring=u"recall_macro", cv=10, n_jobs=-1, verbose=3)

grid_search.fit(X_train, y_train)
# for key, value in grid_search.cv_results_.items():
#     print key, value

predictions = grid_search.predict(X_test)
print u"macro_recall", recall_score(y_test, predictions, average=u"macro")
print precision_recall_fscore_support(y_test, predictions)
print confusion_matrix(y_test, predictions)

from sklearn.externals import joblib

joblib.dump(grid_search.best_estimator_, model_dump_filename)

clf = joblib.load(model_dump_filename)
