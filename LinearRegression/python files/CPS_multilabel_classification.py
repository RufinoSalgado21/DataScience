import os

import keras
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular

# Opens and reads selected dataset, returning dataframe
from matplotlib import pyplot
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def read_file(filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + filename
    dataset = pd.read_csv(path)
    print('Dataset read.')
    return dataset


# Fill missing values with mean values for respective columns
def replace_NaN(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X)
    X = imputer.transform(X)
    '''
    print("Missing value replaced.")
    print(X[0])
    '''
    return X


# Encoding My Voice, My School 5 Essentials Survey  Score column
def encode_X(X):
    transformer = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), ['My Voice, My School 5 Essentials Survey  Score '])],
        remainder='passthrough')
    X = transformer.fit_transform(X)
    '''
    print('X')
    print(X[0])
    print("Encoded.")
    print(X[0])
    '''
    return X


def main():
    # Read dataset
    path = "SQRPratings_2019-2020_SchoolLevel_9to12_edited.csv"
    dataset = read_file(path)

    # Select columns for features and column for dependent variable
    X = dataset.drop(
        columns=['School ID', 'School Name', 'Network ', 'SQRP Total Points Earned', 'SY 2019-2020 SQRP Rating',
                 'SY 2019-2020 Accountability Status'])
    Y = dataset[['SY 2019-2020 SQRP Rating']]

    features = pd.get_dummies(X)
    features = features.columns
    # print(len(features))
    X = encode_X(X)
    for i in range(len(features)):
        print(features[i])
        print(X[0][i])
    # for multiclass, multilabel, we want the dependent to resemble a
    # onehotencoded matrix of 0s and 1s so theres a column for each category
    print(Y)
    Y = pd.get_dummies(Y)
    print(Y)
    X = replace_NaN(X)

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(units=len(X), activation='relu', input_shape=(len(X[0]),)))
    model.add(keras.layers.Dense(units=len(X), activation='relu'))
    model.add(keras.layers.Dense(units=6, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X, Y, batch_size=32, epochs=30)
    # print(Y)

    '''
    pyplot.plot(history.history['accuracy'])
    pyplot.show()
    '''
    # print(Y)
    # input for predictions should be in shape of a row
    output = model.predict(X[0].reshape(1, len(X[0])))
    # run amax on axis bc output is a row
    max = np.amax(output, axis=1)
    #print(output)
    #print(max)

    for i in range(len(output[0])):
        peak = -1
        if output[0][i] > peak:
            peak = i
        result = peak
    #print(result)
    print("Predicted Label: " + str(Y.columns[result-1]))

    '''
    explainer = lime.lime_tabular.LimeTabularExplainer(X,feature_names = features, class_names=[0,1,2,3,4,5])
    x = X[0].reshape(1,len(X[0]))
    xshape=X[0]
    print(x.shape)
    print(xshape.shape)
    exp = explainer.explain_instance(xshape, model.predict_proba, num_features=72, labels=[0,1,2,3,4,5])

    #print(exp.as_list(label=0))
    print('\n'.join(map(str, exp.as_list(label=0))))
    '''


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    np.set_printoptions(threshold=np.inf)
    main()