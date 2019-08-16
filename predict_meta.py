from sklearn.preprocessing import MinMaxScaler
import numpy as np
from joblib import dump, load


Model="knn.joblib"
Scaler="scaler.joblib"

classes=['sklearn.KNeighborsClassifier',
       'sklearn.GaussianProcessClassifier',
       'sklearn.DecisionTreeClassifier', 'sklearn.RandomForestClassifier',
       'sklearn.AdaBoostClassifier', 'sklearn.GaussianNB',
       'sklearn.QuadraticDiscriminantAnalysis',
       'sklearn.GradientBoostingClassifier',
       'sklearn.LinearDiscriminantAnalysis', 'sklearn.Perceptron',
       'sklearn.LogisticRegression', 'sklearn.ComplementNB',
       'sklearn.SVC']


def predict_meta(meta):
    model=load(Model)
    scaler=load(Scaler)
    X=scaler.transform(meta.reshape(1,-1))
    outp=model.predict_proba(X)[0]
    srt=np.argsort(outp)[::-1]
    ress=[[classes[srt[i]][8:],float(outp[srt[i]])] for i in range(len(srt))]
    return ress 

