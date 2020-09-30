import keras
import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_file(filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + filename
    dataset = pd.read_csv(path)
    print('Dataset read.')
    return dataset

#Iterates over features checking for string values
#Returns: An updated list with categorical values encoded
def check_for_encoding(X):
    cols_need_encoding = []
    for i in range(len(X[0])):
        for j in range(len(X[:, i])):
            if isinstance(X[j, i], str):
                #print("Column Needs Encoding:" + str(i))
                cols_need_encoding.append(i)
                break

    for c in cols_need_encoding:
        transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [c])], remainder='passthrough')
        X = np.array(transformer.fit_transform(X[:,:]))
        return X



def main():
    filename = 'SQRPratings_2019-2020_SchoolLevel_9to12_edited.csv'
    dataset = read_file(filename)
    X = dataset.iloc[:,6:].values
    Y = dataset.iloc[:,3].values

    X = check_for_encoding(X)
    print("Encoding")

    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(X[:,:])
    X = imputer.transform(X)
    print("Replacing missing values")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print("Splitting into training and testing sets")

    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    print("Fitting regression model")

    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1),Y_test.reshape(len(Y_test), 1)),1))

    #print(X_train[0])



if __name__ == '__main__':
    main()