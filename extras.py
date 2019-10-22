from autosklearn import metrics

CLASSIFIERS=["adaboost","bernoulli_nb","decision_tree", "extra_trees","gaussian_nb", "gradient_boosting","k_nearest_neighbors", "lda","liblinear_svc","libsvm_svc","multinomial_nb","passive_aggressive","qda","random_forest","sgd"]

CLASSIFIERS_DISP=["AdaBoost","Bernoulli NB","Decision Tree", "Extra Trees","Gaussian NB", "Gradient Boosting","K Nearest Neighbors", "LDA","Liblinear SVC","Libsvm SVC","Multinomial NB","Passive Aggressive","QDA","Random Forest","SGD"]

REGRESSORS=["adaboost","ard_regression","decision_tree", "extra_trees","gaussian_process", "gradient_boosting","k_nearest_neighbors","liblinear_svr","libsvm_svr","random_forest","sgd","xgradient_boosting"]

REGRESSORS_DISP=["AdaBoost","ARD","Decision Tree", "Extra Trees","Gaussian Process", "Gradient Boosting","K Nearest Neighbors","Liblinear SVR","Libsvm SVR","Random Forest","SGD","XGradient Boosting"]

PREPROCESSORS_CL=["no_preprocessing","extra_trees_preproc_for_classification","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

PREPROCESSORS_CL_DISP=["No Preprocessing","Extra Trees Preprocessor","Fast ICA","Feature Agglomeration","Kernel PCA","Litchen Sinks","Liblinear SVC Preprocessor","Nystroem Sampler","PCA","Polynomial","Random Trees Embedding","Select Percentile Classification","Select Percentile Regression","Select Rates","Truncated SVD"]

PREPROCESSORS_RG=["no_preprocessing","extra_trees_preproc_for_regression","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

PREPROCESSORS_RG_DISP=["No Preprocessing","Extra Trees Preprocessor","Fast ICA","Feature Agglomeration","Kernel PCA","Kitchen Sinks","Liblinear SVC Preprocessor","Nystroem Sampler","PCA","Polynomial","Random Trees Embedding","Select Percentile Classification","Select Percentile Regression","Select Rates","Truncated SVD"]

METRICS_CL=[metrics.accuracy,metrics.f1_macro,metrics.precision, metrics.recall]

METRICS_RG=[metrics.r2,metrics.mean_squared_error,metrics.mean_absolute_error,metrics.median_absolute_error]

METRICS_CL_DISP=["Accuracy","F1","Precision","Recall"]

METRICS_RG_DISP=["R2","Mean Squared Error","Mean Absolute Error","Median Absolute Error"]

CLASSIFIERS=["adaboost","bernoulli_nb","decision_tree", "extra_trees","gaussian_nb", "gradient_boosting","k_nearest_neighbors", "lda","liblinear_svc","libsvm_svc","multinomial_nb","passive_aggressive","qda","random_forest","sgd"]



ESTIMATOR_TIMES={'Adjusted SVM': 0.07594684791087697,
         'Linear SVM': 0.036905399905705766,

'liblinear_svc': 0.07594684791087697,
'libsvm_svc': 0.07594684791087697,
'sgd': 0.07594684791087697,
'passive_agressive': 0.07594684791087697,
    

        'random_forest': 0.005704711340354813,
           'k_nearest_neightbors': 0.27603989952945424,
            'decision_tree': 0.0006857563445533376,
             'adaboost': 0.008235442340735408,
              'Naive Bayes': 0.0001938958234472096,
              
              'bernoulli_nb': 0.0001938958234472096,
              'gaussian_nb': 0.0001938958234472096,
              'multinomial_nb': 0.0001938958234472096,
              
              'lda': 0.004516812286156457,
                'qda': 0.005043238462350772,
                 'gradient_boosting': 0.5271939275232157,
                  'logistic_regression': 0.059534068533149326}

def gen_metric(task, metrics_choice):
    if task=="classification":
        return METRICS_CL[METRICS_CL_DISP.index(metrics_choice)]
    else:
        return METRICS_RG[METRICS_RG_DISP.index(metrics_choice)]




def format_ls(ls,val):
    if(ls=="cl"):
        rs = CLASSIFIERS_DISP[CLASSIFIERS.index(val)]
    elif(ls=="rg"):
        rs = REGRESSORS_DISP[REGRESSORS.index(val)]
    elif(ls=="cp"):
        rs = PREPROCESSORS_CL_DISP[PREPROCESSORS_CL.index(val)]
    elif(ls=="rp"):
        rs = PREPROCESSORS_RG_DISP[PREPROCESSORS_RG.index(val)]
    else:
        rs ="wrong argument"
    return rs




def format_time(secs):
    days=(secs//(60*60*24))
    hours=(secs%(60*60*24))//(60*60)
    minutes=(secs%(60*60))//(60)
    seconds=(secs%(60))
    if(minutes<1):
        return str(seconds)+"s "
    elif(hours<1):
        return str(minutes)+"m "+str(seconds)+"s "
    elif(days<1):
        return str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s "
    else:
        return str(days)+"d "+str(hours)+"h "+str(minutes)+"m "+str(seconds)+"s "

