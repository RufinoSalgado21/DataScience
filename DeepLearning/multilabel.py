import os

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# load dataset
path = os.environ['PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + 'iris.csv'
file = open(path)
dataframe = pandas.read_csv(file, header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

'''
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

pred_x = X[0].reshape(1,4)
model = baseline_model()
model.fit(X, dummy_y, batch_size=32, epochs=30)
#prediction = model.predict(pred_x)
labels = ['Iris-setosa','Iris-versicolor','Iris-virginica']
#for i in range(len(prediction[0])):
#	print(labels[i] + ': ' + str(prediction[0][i]))

row = 0
for i in X:
	print('Row: ' + str(row))
	row+=1
	pred_x = i.reshape(1,4)
	prediction = model.predict(pred_x)
	for i in range(len(prediction[0])):
		print(labels[i] + ': ' + str(prediction[0][i]))
	print()