import pandas as pd
import sklearn.impute
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from datetime import datetim

train = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/CaixaBankTech/train.csv')
test = pd.read_csv('https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/CaixaBankTech/test_x.csv')


# se comprueba si hay valores missing
train.isna().sum()
#imputación
train.Close = train.Close.interpolate(method='linear')

X_train = pd.concat([train.Close, train.Close.shift(-3)], axis=1)
X_train = X_train.to_numpy()[:-3, :]
y_train = train.Target.to_numpy()[:-3]
lr = LogisticRegression()
lr.fit(X_train, y_train)
f1_score(lr.predict(X_train), y_train, average='macro') #0.95

X_test = pd.concat([test.Close, test.Close.shift(-3)], axis=1)
X_test.iloc[:, 1] = X_test.iloc[:, 1].interpolate(method="linear")
X_test = X_train.to_numpy()
lr.predict(X_test)

test_final = pd.concat([test.test_index, pd.Series(lr.predict(X_test))], axis=1, names=['test_index', 'Target'])
test_final.to_csv("datos.csv", index=False)