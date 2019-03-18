import pandas as pd
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')


 
train    = pd.read_csv("C:\\Udemy\\Practical Data Science in Python\\Audio\\data_model_neural.csv")
test     = pd.read_csv("C:\\Udemy\\Practical Data Science in Python\\Audio\\validation_neural.csv")

#------------------------------------------------------------------------------------------------#

x = train[["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid",
           "peakf","meanfun","minfun",
       "maxfun","meandom","mindom","maxdom","dfrange","modindx"]]
y = train["class"]

#---------------------------------------------------------------------------#

from keras.utils import np_utils
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


y_train = np_utils.to_categorical(train["y"].values)
y_train = y_train[:,1:6]


model = Sequential()
model.add(Dense(164, input_dim=21, init='uniform', activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(164, activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(40, activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(5, activation='softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',  
              metrics=["accuracy"])  
 
model.fit(x.values, y_train,nb_epoch=7000,verbose=2)

print(model.evaluate(x.values, y_train,verbose=2))



x = test[["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid",
           "peakf","meanfun","minfun",
       "maxfun","meandom","mindom","maxdom","dfrange","modindx"]]



y_test = np_utils.to_categorical(test["y"].values)
y_test = y_test[:,1:6]

print(model.evaluate(x.values, y_test,verbose=2))

