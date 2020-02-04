import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('uploads/blood.csv', sep=',')
features = tpot_data.drop('class', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['class'], random_state=42)

# Average CV score on the training set was: 0.7858241758241757
exported_pipeline = make_pipeline(
    Normalizer(norm="max"),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.2, min_samples_leaf=8, min_samples_split=4, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print(results)
