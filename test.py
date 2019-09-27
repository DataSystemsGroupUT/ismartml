import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier( time_left_for_this_task=30,
            # sec., how long should this seed fit process run
            per_run_time_limit=15,
		ensemble_size=0
)
automl.fit(X_train, y_train)
results=automl.cv_results_
df=pd.DataFrame(data=results).sort_values(by="rank_test_scores")
print(df)
