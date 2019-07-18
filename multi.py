import multiprocessing
import warnings
#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')
import shutil

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.constants import MULTICLASS_CLASSIFICATION
import numpy as np


tmp_folder = 'tmp/autosk_tmp'
output_folder = 'tmp/autosk_out'


for dir_ in [tmp_folder, output_folder]:
    try:
        shutil.rmtree(dir_)
    except OSError:
        pass


def get_spawn_classifier(X_train, y_train):
    def spawn_classifier(seed, time, search_space,prep_space,dataset_name=None):
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
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = AutoSklearnClassifier(
            time_left_for_this_task=30,
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
        automl.fit(X_train, y_train, dataset_name=dataset_name)
        #print(automl.cv_results_)
        return automl.cv_results_
    return spawn_classifier


def get_spawn_regressor(X_train, y_train):
    def spawn_regressor(seed, time,search_space,prep_space,dataset_name=None):
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
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = AutoSklearnRegressor(
            time_left_for_this_task=30,
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
        automl.fit(X_train, y_train, dataset_name=dataset_name)
        #print(automl.cv_results_)
        return automl.cv_results_

    return spawn_regressor

def process_data(path,data_type):
    if(data_type=="numpy"):
        data=np.load(path)
        X=data[:,:-1]
        y=data[:,-1]
    elif(data_type=="csv"):
        data=np.genfromtxt(path,skip_header=1,delimiter=",")
        X=data[:,:-1]
        y=data[:,-1]
    else:
        X=None
        y=None
    return X,y


def run_task(path,task,data_type):

    #interval=time//period
    #extra=time%period
    results=[]
    #data=np.load(path)
    #X=data[:,:-1]
    #y=data[:,-1]
    #X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X,y=process_data(path,data_type)
    
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    
    if task=="classification":
        spawn_estimator = get_spawn_classifier(X_train, y_train)
    elif task=="regression":
        spawn_estimator = get_spawn_regressor(X_train, y_train)

    return spawn_estimator

    """
    for i in range(interval): 
        #results.append(spawn_classifier(i,time))
        results.append(spawn_estimator(i,time))
    if extra >=30:
        results.append(spawn_estimator(i,time))
    return results[-1]
    """

    """
    print('Starting to build an ensemble!')
    automl = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=15,
        ml_memory_limit=1024,
        shared_mode=True,
        ensemble_size=50,
        ensemble_nbest=200,
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        initial_configurations_via_metalearning=0,
        seed=1,
    )

    # Both the ensemble_size and ensemble_nbest parameters can be changed now if
    # necessary
    automl.fit_ensemble(
        y_train,
        task=MULTICLASS_CLASSIFICATION,
        metric=accuracy,
        precision='32',
        dataset_name='digits',
        ensemble_size=20,
        ensemble_nbest=50,
    )

    predictions = automl.predict(X_test)
    print(automl.show_models())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
    """

if __name__ == '__main__':
    #run_task("../digits_c.np.npy","classification",30)
    run_task("../digits_c.np.npy","regression",30)
