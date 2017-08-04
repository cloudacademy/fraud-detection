from __future__ import unicode_literals
import logging
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import (
    recall_score, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from settings import DATASET_FILENAME, MODEL_FILENAME

logger = logging.getLogger('training')

def main():
    """ Find the best model to fit the dataset and save it into file """
    # create a GridSearch object to find the best fitting model
    grid_search = new_grid_search()
    # run the search algorithm
    run_grid_search(grid_search)
    # save the best fitting model into FS
    save_search_results(grid_search)


def split_dataset():
    """ Read and split dataset into train and test subsets """
    df = pd.read_csv(DATASET_FILENAME, header=0)
    X = df[df.columns[:-1]].as_matrix()
    y = df[df.columns[-1]].as_matrix()
    return train_test_split(X, y, test_size=0.2, random_state=42)

def new_grid_search():
    """ Create new GridSearch obj with models pipeline """
    pipeline = Pipeline([
        # TODO some smart preproc can be added here
        (u"clf", LogisticRegression(class_weight="balanced")),
    ])
    search_params = {"clf__C": (1e-4, 1e-2, 1e0, 1e2, 1e4)}
    return GridSearchCV(
        estimator=pipeline,
        param_grid=search_params,
        scoring="recall_macro",
        cv=10,
        n_jobs=-1,
        verbose=3,
    )

def run_grid_search(grid_search, show_evaluation=True):
    """ Run the GridSearch algorithm and compute evaluation metrics """
    X_train, X_test, y_train, y_test = split_dataset()

    grid_search.fit(X_train, y_train)
    # for key, value in grid_search.cv_results_.items():
    #     print key, value

    predictions = grid_search.predict(X_test)

    if show_evaluation:
        logger.debug("macro_recall: %s", recall_score(y_test, predictions, average="macro"))
        logger.debug(precision_recall_fscore_support(y_test, predictions))
        logger.debug(confusion_matrix(y_test, predictions))

def save_search_results(grid_search):
    """ Serialize model into file """
    joblib.dump(grid_search.best_estimator_, MODEL_FILENAME)
    # then load it like this:
    # clf = joblib.load(model_dump_filename)

if __name__ == "__main__":
    main()
