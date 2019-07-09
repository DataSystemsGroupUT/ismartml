import numpy as np
import warnings
#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')


import autosklearn.classification
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

def run_task(path,task, time):
    results=[]
    data=np.load(path)
    X=data[:,:-1]
    y=data[:,-1]
    #X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
    if task == "classification": 
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time,
            per_run_time_limit=1
        )
    elif task == "regression":
        automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=time,
                per_run_time_limit=1,
            )
    else:
        return None

        #predictions = automl.predict(X_test)
        #print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))


    automl.fit(X_train, y_train)
    print("aaaaa")
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    #results.append("Accuracy score: "+ str(sklearn.metrics.accuracy_score(y_test, y_hat)))
    results.append(None)
    #results.append(automl.show_models())
    results.append(automl.cv_results_)
    #results.append(automl.sprint_statistics())
    #print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))
    return results



if __name__=="__main__":
    #print(classification_task("../digits_c.np.npy"))
    run_task("../digits_c.np.npy","classification",30)

