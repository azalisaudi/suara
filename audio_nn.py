import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

df = pd.read_csv("Audio/data2.csv")

x = df[["meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","skew","kurt","sp.ent","sfm","meandom","mindom","maxdom","dfrange","modindx","dfslope","meanpeakf"]]

LABEL = 'encoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['class'].values.ravel())
num_classes = le.classes_.size

y = df["encoded"]
# Convert to One Hot Encoding
y = np_utils.to_categorical(y, num_classes)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = Sequential()
model.add(Dense(128, input_dim=17, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.05))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])  
model.fit(X_train, y_train, epochs=5000, verbose=1)

print("\nTraining score")
train_score = model.evaluate(X_train, y_train, verbose=1)
print("Train accuracy ", train_score[1])

print("\nTest score")
test_score = model.evaluate(X_test, y_test, verbose=1)
print("Test accuracy ", test_score[1])

max_y_test  = np.argmax(y_test, axis=1)
max_y_pred  = np.argmax(model.predict(X_test), axis=1)
print("\nConfusion Matrix")
print(confusion_matrix(max_y_test, max_y_pred))
print("\nClassification Report")
print(classification_report(max_y_test, max_y_pred))
