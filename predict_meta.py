from sklearn.preprocessing import MinMaxScaler
import numpy as np
from joblib import dump, load


#Model="knn.joblib"
Model="fr.joblib"
Scaler="scaler.joblib"


Time_Model="adaboost.joblib"

classes_og=['sklearn.KNeighborsClassifier',
       'sklearn.GaussianProcessClassifier',
       'sklearn.DecisionTreeClassifier', 'sklearn.RandomForestClassifier',
       'sklearn.AdaBoostClassifier', 'sklearn.GaussianNB',
       'sklearn.QuadraticDiscriminantAnalysis',
       'sklearn.GradientBoostingClassifier',
       'sklearn.LinearDiscriminantAnalysis', 'sklearn.Perceptron',
       'sklearn.LogisticRegression', 'sklearn.ComplementNB',
       'sklearn.SVC']

classes=['K Nearest Neighbors',
       'Gaussian Process',
       'Decision Tree', 'Random Forest',
       'AdaBoost', 'Gaussian NB',
       'QDA',
       'Gradient Boosting',
       'LDA', 'Perceptron',
       'Logistic Regression', 'Complement NB',
       'SVC']



excluded=['Gaussian Process','Perceptron', 'Logistic Regression', 'Complement NB']

def filter_excluded(ls):
    res=[]
    for ent in ls:
        if ent[0] in excluded:
            res.append([ent[0],float(0)])
        else:
            res.append(ent)
    return res

def predict_meta(meta):
    model=load(Model)
    scaler=load(Scaler)
    X=scaler.transform(meta.reshape(1,-1))
    outp=model.predict_proba(X)[0]
    srt=np.argsort(outp)[::-1]
    ress=[[classes[srt[i]],float(outp[srt[i]])] for i in range(len(srt))]
    ress=filter_excluded(ress) #comment this out to get full results
    return ress 

def predict_time(meta):

    #model=load(Time_Model)
    #X=meta[[0,2]].reshape(1,-1)
    #outp=model.predict(X)
    #print(meta)
    print(meta[0],meta[2])
    outp=meta[0]*meta[2]/200
    return outp

