import numpy as np
import warnings
#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')


import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

def classification_task(path):
    results=[]
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
    results.append("Accuracy score: "+ str(sklearn.metrics.accuracy_score(y_test, y_hat)))
    #results.append(automl.show_models())
    results.append(automl.cv_results_)
    #results.append(automl.sprint_statistics())
    #print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    return results

if __name__=="__main__":
    print(classification_task("../digits_c.np.npy"))

