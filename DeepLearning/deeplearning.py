import os

import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot


def read_file(filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + filename
    dataset = pd.read_csv(path)
    print('Dataset read.')
    return dataset

filename = 'HousingPrices.csv'
dataset = read_file(filename)

print(dataset)

X = dataset.drop(columns = ['SalePrice'])
Y = dataset[['SalePrice']]
print(Y)

model = keras.models.Sequential()

model.add(keras.layers.Dense(units=8, activation='relu', input_shape=(8,)))
model.add(keras.layers.Dense(units=8, activation='relu'))
model.add(keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine'])

history = model.fit(X,Y, epochs=30 )

# plot metrics
pyplot.plot(history.history['mse'])
pyplot.plot(history.history['mae'])
pyplot.plot(history.history['mape'])
pyplot.plot(history.history['cosine'])
pyplot.show()

test_data = np.array([2003,	854,	1710,	2,	1,	3,	8,	2008])
print(model.predict(test_data.reshape(1,8), batch_size=1))
