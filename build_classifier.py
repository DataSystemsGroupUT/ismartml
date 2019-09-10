#!/usr/bin/env python
# coding: utf-8

# In[239]:


from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import PassiveAggressiveClassifier,SGDClassifier


# In[240]:


def adaboost_cl(params):
    max_depth=params.pop("max_depth")
    params["base_estimator"]="DecisionTreeClassifier(max_depth={})".format(max_depth)
    cl=AdaBoostClassifier(**params)
    return cl


# In[241]:


def decision_tree_cl(params):
    max_depth=params.pop("max_depth_factor")
    cl=DecisionTreeClassifier(**params)
    return cl


# In[242]:


def bernoulli_nb_cl(params):
    #max_depth=params.pop("max_depth_factor")
    cl=BernoulliNB(**params)
    return cl


# In[243]:


def extra_trees_cl(params):
    cl=ExtraTreesClassifier(**params)
    return cl


# In[244]:


def gaussian_nb_cl(params):
    cl=GaussianNB(**params)
    return cl


# In[245]:


def gradient_boosting_cl(params):
    cl=GradientBoostingClassifier(**params)
    return cl


# In[246]:


def k_nearest_neighbors_cl(params):
    cl=KNeighborsClassifier(**params)
    return cl


# In[247]:


def lda_cl(params):
    cl=LinearDiscriminantAnalysis(**params)
    return cl


# In[248]:


def liblinear_svc_cl(params):
    cl=LinearSVC(**params)
    return cl


# In[249]:


def libsvm_svc_cl(params):
    cl=SVC(**params)
    return cl


# In[250]:


def qda_cl(params):
    cl=QuadraticDiscriminantAnalysis(**params)
    return cl


# In[251]:


def passive_aggressive_cl(params):
    cl=PassiveAggressiveClassifier(**params)
    return cl


# In[252]:


def multinomial_nb_cl(params):
    cl=MultinomialNB(**params)
    return cl


# In[253]:


def random_forest_cl(params):
    cl=RandomForestClassifier(**params)
    return cl


# In[254]:


def sgd_cl(params):
    cl=SGDClassifier(**params)
    return cl


# In[259]:


res=[0.0, {'balancing:strategy': 'none', 'categorical_encoding:__choice__': 'one_hot_encoding', 'classifier:__choice__': 'xgradient_boosting', 'imputation:strategy': 'mean', 'preprocessor:__choice__': 'no_preprocessing', 'rescaling:__choice__': 'standardize', 'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'True', 'classifier:xgradient_boosting:base_score': 0.5, 'classifier:xgradient_boosting:booster': 'gbtree', 'classifier:xgradient_boosting:colsample_bylevel': 1.0, 'classifier:xgradient_boosting:colsample_bytree': 1.0, 'classifier:xgradient_boosting:gamma': 0, 'classifier:xgradient_boosting:learning_rate': 0.1, 'classifier:xgradient_boosting:max_delta_step': 0, 'classifier:xgradient_boosting:max_depth': 3, 'classifier:xgradient_boosting:min_child_weight': 1, 'classifier:xgradient_boosting:n_estimators': 512, 'classifier:xgradient_boosting:reg_alpha': 1e-10, 'classifier:xgradient_boosting:reg_lambda': 1e-10, 'classifier:xgradient_boosting:scale_pos_weight': 1, 'classifier:xgradient_boosting:subsample': 1.0, 'categorical_encoding:one_hot_encoding:minimum_fraction': 0.01}]


# In[256]:


a=["adaboost","bernoulli_nb","decision_tree", "extra_trees","gaussian_nb", "gradient_boosting","k_nearest_neighbors", "lda","liblinear_svc","libsvm_svc","multinomial_nb","passive_aggressive","qda","random_forest","sgd","xgradient_boosting"]


# In[257]:


def build_classifier(param_dict):
    print(param_dict["classifier:__choice__"])
    params={}
    pre="classifier:{}:".format(param_dict["classifier:__choice__"])
    for key in param_dict.keys():
        if pre in key:
            params[key[len(pre):]]=param_dict[key]
    if param_dict["classifier:__choice__"]=="adaboost":
            return adaboost_cl(params)
    elif param_dict["classifier:__choice__"]=="decision_tree":
            return decision_tree_cl(params)
    elif param_dict["classifier:__choice__"]=="bernoulli_nb":
            return bernoulli_nb_cl(params)
    elif param_dict["classifier:__choice__"]=="extra_trees":
            return extra_trees_cl(params)
    elif param_dict["classifier:__choice__"]=="gaussian_nb":
            return gaussian_nb_cl(params)
    elif param_dict["classifier:__choice__"]=="gradient_boosting":
            return gradient_boosting_cl(params)
    elif param_dict["classifier:__choice__"]=="k_nearest_neighbors":
            return k_nearest_neighbors_cl(params)
    elif param_dict["classifier:__choice__"]=="lda":
            return lda_cl(params)
    elif param_dict["classifier:__choice__"]=="liblinear_svc":
            return liblinear_svc_cl(params)
    elif param_dict["classifier:__choice__"]=="libsvm_svc":
            return libsvm_svc_cl(params)
    elif param_dict["classifier:__choice__"]=="qda":
        return qda_cl(params)
    elif param_dict["classifier:__choice__"]=="passive_aggressive":
        return passive_aggressive_cl(params)
    elif param_dict["classifier:__choice__"]=="multinomial_nb":
        return multinomial_nb_cl(params)
    elif param_dict["classifier:__choice__"]=="random_forest":
        return random_forest_cl(params)
    elif param_dict["classifier:__choice__"]=="sgd":
        return sgd_cl(params)
    else:
        return None


    

