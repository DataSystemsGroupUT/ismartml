

CLASSIFIERS=["adaboost","bernoulli_nb","decision_tree", "extra_trees","gaussian_nb", "gradient_boosting","k_nearest_neighbors", "lda","liblinear_svc","libsvm_svc","multinomial_nb","passive_aggressive","qda","random_forest","sgd","xgradient_boosting"]

CLASSIFIERS_DISP=["AdaBoost","Bernoulli NB","Decision Tree", "Extra Trees","Gaussian NB", "Gradient Boosting","K Nearest Neighbors", "LDA","Liblinear SVC","Libsvm SVC","Multinomial NB","Passive Aggressive","QDA","Random Forest","SGD","XGradient Boosting"]

REGRESSORS=["adaboost","ard_regression","decision_tree", "extra_trees","gaussian_process", "gradient_boosting","k_nearest_neighbors","liblinear_svr","libsvm_svr","random_forest","sgd","xgradient_boosting"]

REGRESSORS_DISP=["AdaBoost","ARD","Decision Tree", "Extra Trees","Gaussian Process", "Gradient Boosting","K Nearest Neighbors","Liblinear SVR","Libsvm SVR","Random Forest","SGD","XGradient Boosting"]

PREPROCESSORS_CL=["no_preprocessing","extra_trees_preproc_for_classification","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

PREPROCESSORS_CL_DISP=["No Preprocessing","Extra Trees Preprocessor","Fast ICA","Feature Agglomeration","Kernel PCA","Litchen Sinks","Liblinear SVC Preprocessor","Nystroem Sampler","PCA","Polynomial","Random Trees Embedding","Select Percentile Classification","Select Percentile Regression","Select Rates","Truncated SVD"]

PREPROCESSORS_RG=["no_preprocessing","extra_trees_preproc_for_regression","fast_ica","feature_agglomeration","kernel_pca","kitchen_sinks","liblinear_svc_preprocessor","nystroem_sampler","pca","polynomial","random_trees_embedding","select_percentile_classification","select_percentile_regression","select_rates","truncatedSVD"]

PREPROCESSORS_RG_DISP=["No Preprocessing","Extra Trees Preprocessor","Fast ICA","Feature Agglomeration","Kernel PCA","Kitchen Sinks","Liblinear SVC Preprocessor","Nystroem Sampler","PCA","Polynomial","Random Trees Embedding","Select Percentile Classification","Select Percentile Regression","Select Rates","Truncated SVD"]

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

