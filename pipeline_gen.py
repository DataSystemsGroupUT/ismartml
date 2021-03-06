#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[45]:


from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from autosklearn.pipeline.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC


# In[33]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[34]:


def fast_ica(params):
    pr = FastICA(**params)
    return pr


def extra_trees_preproc_for_classification(params):
    pr = ExtraTreesClassifier(**params)
    return pr


def no_preprocessing(params):
    pr = None
    return pr


def liblinear_svc_preprocessor(params):
    # pr=LinearSVC(**params)
    pr = LibLinear_Preprocessor(**params)
    return pr


def polynomial(params):
    pr = PolynomialFeatures(**params)
    return pr


def random_trees_embedding(params):
    pr = RandomTreesEmbedding(**params)
    return pr


def pca(params):
    params.pop("keep_variance")
    pr = PCA(**params)
    return pr


def pca(params):
    params.pop("keep_variance")
    pr = PCA(**params)
    return pr


def build_preprocessor_cl(param_dict):
    print(param_dict["preprocessor:__choice__"])
    params = {}
    pre = "preprocessor:{}:".format(param_dict["preprocessor:__choice__"])
    for key in param_dict.keys():
        if pre in key:
            params[key[len(pre):]] = param_dict[key]
    if param_dict["preprocessor:__choice__"] == "fast_ica":
        return fast_ica(params)
    elif param_dict["preprocessor:__choice__"] == "extra_trees_preproc_for_classification":
        return extra_trees_preproc_for_classification(params)
    elif param_dict["preprocessor:__choice__"] == "no_preprocessing":
        return no_preprocessing(params)
    elif param_dict["preprocessor:__choice__"] == "liblinear_svc_preprocessor":
        return liblinear_svc_preprocessor(params)
    elif param_dict["preprocessor:__choice__"] == "random_trees_embedding":
        return random_trees_embedding(params)
    elif param_dict["preprocessor:__choice__"] == "polynomial":
        return polynomial(params)
    elif param_dict["preprocessor:__choice__"] == "pca":
        return pca(params)
    # elif param_dict["preprocessor:__choice__"]=="pca":
    #        return pca(params)


# In[35]:


def adaboost_cl(params):
    max_depth = params.pop("max_depth")
    params["base_estimator"] = DecisionTreeClassifier(max_depth=max_depth)
    cl = AdaBoostClassifier(**params)
    return cl


def decision_tree_cl(params):
    max_depth = params.pop("max_depth_factor")
    cl = DecisionTreeClassifier(**params)
    return cl


def bernoulli_nb_cl(params):
    cl = BernoulliNB(**params)
    return cl


def extra_trees_cl(params):
    cl = ExtraTreesClassifier(**params)
    return cl


def gaussian_nb_cl(params):
    cl = GaussianNB(**params)
    return cl


def gradient_boosting_cl(params):
    cl = GradientBoostingClassifier(**params)
    return cl


def k_nearest_neighbors_cl(params):
    cl = KNeighborsClassifier(**params)
    return cl


def lda_cl(params):
    cl = LinearDiscriminantAnalysis(**params)
    return cl


def liblinear_svc_cl(params):
    # cl=LinearSVC(**params)
    cl = LibLinear_SVC(**params)
    return cl


def libsvm_svc_cl(params):
    # cl=SVC(**params)
    cl = LibSVM_SVC(**params)
    return cl


def qda_cl(params):
    cl = QuadraticDiscriminantAnalysis(**params)
    return cl


def passive_aggressive_cl(params):
    cl = PassiveAggressiveClassifier(**params)
    return cl


def multinomial_nb_cl(params):
    cl = MultinomialNB(**params)
    return cl


def random_forest_cl(params):
    cl = RandomForestClassifier(**params)
    return cl


def sgd_cl(params):
    cl = SGDClassifier(**params)
    return cl


def build_classifier(param_dict):
    print(param_dict["classifier:__choice__"])
    params = {}
    pre = "classifier:{}:".format(param_dict["classifier:__choice__"])
    for key in param_dict.keys():
        if pre in key:
            params[key[len(pre):]] = param_dict[key]
    if param_dict["classifier:__choice__"] == "adaboost":
        return adaboost_cl(params)
    elif param_dict["classifier:__choice__"] == "decision_tree":
        return decision_tree_cl(params)
    elif param_dict["classifier:__choice__"] == "bernoulli_nb":
        return bernoulli_nb_cl(params)
    elif param_dict["classifier:__choice__"] == "extra_trees":
        return extra_trees_cl(params)
    elif param_dict["classifier:__choice__"] == "gaussian_nb":
        return gaussian_nb_cl(params)
    elif param_dict["classifier:__choice__"] == "gradient_boosting":
        return gradient_boosting_cl(params)
    elif param_dict["classifier:__choice__"] == "k_nearest_neighbors":
        return k_nearest_neighbors_cl(params)
    elif param_dict["classifier:__choice__"] == "lda":
        return lda_cl(params)
    elif param_dict["classifier:__choice__"] == "liblinear_svc":
        return liblinear_svc_cl(params)
    elif param_dict["classifier:__choice__"] == "libsvm_svc":
        return libsvm_svc_cl(params)
    elif param_dict["classifier:__choice__"] == "qda":
        return qda_cl(params)
    elif param_dict["classifier:__choice__"] == "passive_aggressive":
        return passive_aggressive_cl(params)
    elif param_dict["classifier:__choice__"] == "multinomial_nb":
        return multinomial_nb_cl(params)
    elif param_dict["classifier:__choice__"] == "random_forest":
        return random_forest_cl(params)
    elif param_dict["classifier:__choice__"] == "sgd":
        return sgd_cl(params)
    else:
        return None


# In[36]:


def process_dict(dict):
    for key in dict.keys():
        if dict[key] == "None":
            dict[key] = None
    return dict


def get_importance(pipe, cl, smote):
    trees = [
        "gradient_boosting",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "adaboost"]
    linear = ["libsvm_svc"]
    ind = 2
    if smote == "yes":
        ind += 1
    if cl in trees:
        return pipe.steps[ind][1].feature_importances_
    if cl in linear:
        try:
            return pipe.steps[ind][1].feature_importances_
        except BaseException:
            pass
    return []


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Confusion matrix, with normalization'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig("static/images/figs/conf_mt_" +
                str(int(normalize)), bbox_inches="tight", transparent=True)
    return ax


def get_matrix(pipe, X, y, smote):
    trees = [
        "gradient_boosting",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "adaboost"]
    linear = ["libsvm_svc"]
    ind = 2
    if smote == "yes":
        ind += 1
    pred_y = pipe.predict(X)
    plot_confusion_matrix(y, pred_y, np.unique(y), normalize=False)
    plot_confusion_matrix(y, pred_y, np.unique(y), normalize=True)
    a = accuracy_score(y, pred_y)
    r = recall_score(y, pred_y, average="macro")
    f = f1_score(y, pred_y, average="macro")
    p = precision_score(y, pred_y, average="macro")
    return [a, r, p, f]


# In[37]:


#res=[0.7976878612716763, {'balancing:strategy': 'none', 'categorical_encoding:__choice__': 'one_hot_encoding', 'classifier:__choice__': 'decision_tree', 'imputation:strategy': 'mean', 'preprocessor:__choice__': 'pca', 'rescaling:__choice__': 'standardize', 'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'False', 'classifier:decision_tree:criterion': 'entropy', 'classifier:decision_tree:max_depth_factor': 1.18671200497328, 'classifier:decision_tree:max_features': 1.0, 'classifier:decision_tree:max_leaf_nodes': 'None', 'classifier:decision_tree:min_impurity_decrease': 0.0, 'classifier:decision_tree:min_samples_leaf': 1, 'classifier:decision_tree:min_samples_split': 2, 'classifier:decision_tree:min_weight_fraction_leaf': 0.0, 'preprocessor:pca:keep_variance': 0.9572746131543354, 'preprocessor:pca:whiten': 'True'}]
#res=[0.7630057803468208, {'balancing:strategy': 'none', 'categorical_encoding:__choice__': 'one_hot_encoding', 'classifier:__choice__': 'adaboost', 'imputation:strategy': 'mean', 'preprocessor:__choice__': 'fast_ica', 'rescaling:__choice__': 'robust_scaler', 'categorical_encoding:one_hot_encoding:use_minimum_fraction': 'False', 'classifier:adaboost:algorithm': 'SAMME', 'classifier:adaboost:learning_rate': 0.4391375941344922, 'classifier:adaboost:max_depth': 3, 'classifier:adaboost:n_estimators': 386, 'preprocessor:fast_ica:algorithm': 'deflation', 'preprocessor:fast_ica:fun': 'cube', 'preprocessor:fast_ica:whiten': 'False', 'rescaling:robust_scaler:q_max': 0.7439738358430176, 'rescaling:robust_scaler:q_min': 0.20581080574615793}]


# """
# pipeline_obj = Pipeline([
#     ('scaler', StandardScaler()),
#     ('svm',SVC())
# ])
# """
# param_dict=process_dict(res[1])
# pipe=Pipeline(([("preprocessor",build_preprocessor_cl(param_dict)),("classifeir",build_classifier(param_dict))]))
# dt=pd.read_csv("blood.csv")
# features=dt.columns[1:-1]
# target=dt.columns[-1]
# X=dt[features]
# y=dt[target]
# pipe.fit(X,y)
# from nyoka import skl_to_pmml
#
#
# #features=[""]
# #target=[""]
# skl_to_pmml(pipe,features,target,"svc_pmml.pmml")

# In[ ]:
