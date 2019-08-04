from sklearn.preprocessing import MinMaxScaler
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
    return classes[outp[0]][8:]

