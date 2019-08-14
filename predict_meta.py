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
    outp=model.predict(X)
    srt=np.argsort(model.predict_proba(X)[0])[::-1]
    ress=[srt,[classes[i][8:] for i in srt]]
    return ress 

