import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



df = pd.read_csv("Audio/data2.csv")

x = df[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","meandom","mindom","maxdom","dfrange","modindx","dfslope","meanpeakf"]]
y = df["class"]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clh = tree.DecisionTreeClassifier(max_depth  = 7)
clf = AdaBoostClassifier(base_estimator= clh,n_estimators=10)
clf.fit(X_train, y_train)


# Display confusion matrix of training data and validation score
print("-------------------------------------------")
print("---------------- Adabost ------------------")
print("-------------------------------------------")
print("Accuracy of training data")
pred = clf.predict(X_train)
cf = confusion_matrix(y_train, pred)
print(cf)
cv_score = np.mean(cross_val_score(clf, X_train, y_train, cv=8))
print("Cross validation score: ", cv_score)


print("\nAccuracy of testing data")
pred = clf.predict(X_test)
cf = confusion_matrix(y_test, pred)
print(cf)
test_score = clf.score(X_test, y_test)
print("Test score: ", test_score)
