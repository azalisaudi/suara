from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# Set some standard parameters upfront
pd.options.display.float_format = '{:.6f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')

print('keras version ', keras.__version__)

# Same labels will be reused throughout the program
LABELS = ['Banana',
          'Chair',
          'Goodbye',
          'Hello',
          'IceCream']

data = pd.read_csv("Audio/data.csv")

x = data[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","dfslope","meanpeakf"]]

LABEL = 'y'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
data[LABEL] = le.fit_transform(data['class'].values.ravel())
print(data)

num_classes = le.classes_.size
print(list(le.classes_))

y = data["y"]
print(y)

# Convert to One Hot Encoding
y = np_utils.to_categorical(y, num_classes)
print(y)

# Splitting the dataset into the Training set and Test setfrom sklearn.model_selection import train_test_split
# 80% Training, 20% Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = Sequential()
model.add(Dense(164, input_dim=20, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(164, activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(40, activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(5, activation='softmax'))
print(model.summary())
 
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])  

NUM_EPOCHS = 7000 
history = model.fit(x_train.values, y_train, epochs=NUM_EPOCHS, verbose=1)

score = model.evaluate(x_train.values, y_train, verbose=1)
accuracy = score[1] * 100
loss = score[0] * 100
print('Accuracy on train data: %0.2f' % accuracy)
print('Loss on train data: %0.2f' % loss)

score = model.evaluate(x_test.values, y_test, verbose=1)
accuracy = score[1] * 100
loss = score[0] * 100
print('Accuracy on test data: %0.2f' % accuracy)
print('Loss on test data: %0.2f' % loss)
