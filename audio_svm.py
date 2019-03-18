import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC,LinearSVC,NuSVC
 
df = pd.read_csv("Audio/data2.csv")

x = df[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","meandom","mindom","maxdom","dfrange","modindx","dfslope","meanpeakf"]]
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svcfit  = SVC(C=0.01, kernel='linear')
X_train =  preprocessing.scale(X_train)
X_test  =  preprocessing.scale(X_test)

svcfit.fit(X_train, y_train)

# Display confusion matrix of training data and validation score
print("---------------------------------------")
print("---------------- SVM ------------------")
print("---------------------------------------")

print("Accuracy of training data")
pred = svcfit.predict(X_train)
cf = confusion_matrix(y_train, pred)
print(cf)
cv_score = np.mean(cross_val_score(svcfit, X_train, y_train, cv=8))
print("Cross validation score: ", cv_score)

print("\nAccuracy of testing data")
pred = svcfit.predict(X_test)
cf = confusion_matrix(y_test, pred)
print(cf)
test_score = svcfit.score(X_test, y_test)
print("Test score: ", test_score)
