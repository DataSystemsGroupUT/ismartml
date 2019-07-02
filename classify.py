import numpy as np
import warnings
#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')


import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

def classification_task(path):
    data=np.load(path)
    X=data[:,:-1]
    y=data[:,-1]
    #X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=1
    )
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    return "Accuracy score: "+ str(sklearn.metrics.accuracy_score(y_test, y_hat))

if __name__=="__main__":
    print(classification_task("../digits_c.np.npy"))

