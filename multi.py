""" Running AutoSklearn on the data """
import shutil
import warnings
import numpy as np
import pandas as pd
import sklearn.model_selection
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from tpot import TPOTClassifier

warnings.filterwarnings('ignore')


tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'


for dir_ in [tmp_folder, output_folder]:
    try:
        shutil.rmtree(dir_)
    except OSError:
        pass


def get_spawn_classifier(X_train, y_train, X_test=None, y_test=None):
    """Generates and returns spaw_classifier """
    def spawn_classifier(
            seed,
            time,
            search_space,
            prep_space,
            metric,
            dataset_name=None):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only in one out of
        # the four processes spawned. This prevents auto-sklearn from evaluating
        # the same configurations in four processes.
        if seed == 0:
            initial_configurations_via_metalearning = 25
            smac_scenario_args = {}
        else:
            initial_configurations_via_metalearning = 0
            smac_scenario_args = {'initial_incumbent': 'RANDOM'}

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different
        # seed!
        automl = AutoSklearnClassifier(
            time_left_for_this_task=time,
            # sec., how long should this seed fit process run
            per_run_time_limit=15,
            # sec., each model may only take this long before it's killed
            ml_memory_limit=1024,
            # MB, memory limit imposed on each call to a ML algorithm
            shared_mode=True,  # tmp folder will be shared between seeds
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=False,
            ensemble_size=0,
            include_estimators=search_space, exclude_estimators=None,
            include_preprocessors=prep_space, exclude_preprocessors=None,
            # ensembles will be built when all optimization runs are finished
            initial_configurations_via_metalearning=(
                initial_configurations_via_metalearning
            ),
            seed=seed,
            smac_scenario_args=smac_scenario_args,
        )
        automl.fit(X_train, y_train, X_test=X_test, y_test=y_test,
                   metric=metric, dataset_name=dataset_name)
        # print(automl.cv_results_)
        return automl.cv_results_
    return spawn_classifier


def get_spawn_regressor(X_train, y_train, X_test=None, y_test=None):
    def spawn_regressor(
            seed,
            time,
            search_space,
            prep_space,
            metric,
            dataset_name=None):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only in one out of
        # the four processes spawned. This prevents auto-sklearn from evaluating
        # the same configurations in four processes.
        if seed == 0:
            initial_configurations_via_metalearning = 25
            smac_scenario_args = {}
        else:
            initial_configurations_via_metalearning = 0
            smac_scenario_args = {'initial_incumbent': 'RANDOM'}

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different
        # seed!
        automl = AutoSklearnRegressor(
            time_left_for_this_task=time,
            # sec., how long should this seed fit process run
            per_run_time_limit=15,
            # sec., each model may only take this long before it's killed
            ml_memory_limit=1024,
            # MB, memory limit imposed on each call to a ML algorithm
            shared_mode=True,  # tmp folder will be shared between seeds
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            delete_tmp_folder_after_terminate=False,
            ensemble_size=0,
            include_estimators=search_space, exclude_estimators=None,
            include_preprocessors=prep_space, exclude_preprocessors=None,
            # ensembles will be built when all optimization runs are finished
            initial_configurations_via_metalearning=(
                initial_configurations_via_metalearning
            ),
            seed=seed,
            smac_scenario_args=smac_scenario_args,
        )
        automl.fit(X_train, y_train, X_test=X_test, y_test=y_test,
                   metric=metric, dataset_name=dataset_name)
        # print(automl.cv_results_)
        return automl.cv_results_
    return spawn_regressor


def process_data(path, data_type, target_ft):
    """Loads data and returns as X,y """
    if data_type == "numpy":
        data = np.load(path)
        X = data[:, :-1]
        y = data[:, -1]
    elif data_type == "csv":
        data = pd.read_csv(path)
        X = data.loc[:, data.columns != target_ft].to_numpy()
        y = data.loc[:, target_ft].to_numpy()
        # print(data.columns)
        #print(X.shape, y.shape)
    else:
        X = None
        y = None
    return X, y, data


def run_task(path, task, data_type, target_ft):
    """Runs AutoSklearn optimizer on passed data and parameters """
    X, y, _ = process_data(path, data_type, target_ft)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=1)
    if task == "classification":
        spawn_estimator = get_spawn_classifier(
            X_train, y_train, X_test=X_test, y_test=y_test)
    elif task == "regression":
        spawn_estimator = get_spawn_regressor(
            X_train, y_train, X_test=X_test, y_test=y_test)
    return spawn_estimator

def run_task_tpot(path, task, data_type, time, target_ft):
    """Runs AutoSklearn optimizer on passed data and parameters """
    X, y, _ = process_data(path, data_type, target_ft)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=1)
    if task == "classification":
        pipeline_optimizer = TPOTClassifier(max_time_mins=time, population_size=20, cv=5,
                                    random_state=42, verbosity=2,periodic_checkpoint_folder='tmp/tpot_per',warm_start=True)
    #elif task == "regression":
    #    spawn_estimator = get_spawn_regressor(
    #        X_train, y_train, X_test=X_test, y_test=y_test)
    pipeline_optimizer.fit(X_train, y_train)
    return pipeline_optimizer

