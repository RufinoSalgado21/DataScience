import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
import os

path = os.environ['PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + 'Churn_Modelling.csv'
dataset = pd.read_csv(path)
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = keras.models.Sequential()

ann.add(keras.layers.Dense(units=6, activation='relu'))

ann.add(keras.layers.Dense(units=6, activation='relu'))

ann.add(keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 30)

#1 = exited, 0 = did not exit
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
'''
weights = ann.layers[0].get_weights()[0]
biases = ann.layers[0].get_weights()[1]
print(weights)
print(len(weights))
print(biases)
print(len(biases))

influencesMax = []
influencesMin = []
for i in range(len(weights[0])):
    max = -5
    max_j = -1
    min = 5
    min_j = -1
    for j in range(len(weights[:,i])):
        if weights[j,i] > max:
            max = weights[j,i]
            max_j = j
        if weights[j,i] < min:
            min = weights[j,i]
            min_j = j

    influencesMax.append(max_j)
    influencesMin.append(min_j)

print(influencesMax)
print(influencesMin)
'''