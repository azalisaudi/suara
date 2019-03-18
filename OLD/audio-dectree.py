import pandas as pd
import numpy as np

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn, numpy
    fxn()
    
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
#from sklearn import cross_validation
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC,LinearSVC,NuSVC


    
 
train    = pd.read_csv("Audio/data_model.csv")
test     = pd.read_csv("Audio/validation.csv")

#------------------------------------------------------------------------------------------------#

# sound.files	selec	duration	meanfreq	sd	freq.median	freq.Q25	freq.Q75	freq.IQR	
# time.median	time.Q25	time.Q75	time.IQR	skew	kurt	sp.ent	time.ent	entropy	
# sfm	meanfun	minfun	maxfun	meandom	mindom	maxdom	dfrange	modindx	startdom	enddom	
# dfslope	meanpeakf	class

#x = train[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","mode",
#"centroid","peakf","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]]

x = train[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","dfslope","meanpeakf"]]
y = train["class"]
clh = tree.DecisionTreeClassifier(max_depth  = 7)
clf = AdaBoostClassifier(base_estimator= clh,n_estimators=10)
clf.fit(x,y)

#print(np.mean(cross_validation.cross_val_score(clf, x, y, cv=8)))
###print(np.mean(model_selection.cross_val_score(clf, x, y, cv=8)))
###print(confusion_matrix(y, clf.predict(x)))



#test_x = test[["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode",
#               "centroid","peakf","meanfun","minfun",
#       "maxfun","meandom","mindom","maxdom","dfrange","modindx"]]


test_x = test[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","dfslope","meanpeakf"]]
print("DTA: " + str(clf.score(test_x,test["class"])))

#---------------------------------------------------------------------------#

svcfit = SVC(C=0.01, kernel='linear')
x     =  preprocessing.scale(x)

svcfit.fit(x, y)

#print(np.mean(cross_validation.cross_val_score(svcfit, x, y, cv=8)))
####print(np.mean(model_selection.cross_val_score(svcfit, x, y, cv=8)))
####print(confusion_matrix(y, svcfit.predict(x)))
test_x     =  preprocessing.scale(test_x)
print("SVM: " + str(svcfit.score(test_x,test["class"])))

#---------------------------------------------------------------------------#
