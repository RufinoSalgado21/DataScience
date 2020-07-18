'''
    Rufino Salgado
    June 22, 2020

'''

import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


def import_file(filename):
    pd.set_option('display.max_columns', None)
    path = os.environ[
               'PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + filename
    dataset = pd.read_csv(path, skiprows=[0])
    return dataset


def encode_columns(x, need_of_encode):
    for i in need_of_encode:
        transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [i])], remainder="passthrough")
        x = np.array(transformer.fit_transform(x))
    return x


def fill_missing_values(X, start, end):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(X[:, start:end])
    X[:, start:end-4] = imputer.transform(X[:, start:end])


def split_variables(data):
    X = data.iloc[1:-3, 4:-85].values
    Y = data.iloc[1:-3, 3].values
    print('X[0] len')
    print(len(X[0]))
    print('Y[0] len')
    print(len(Y))
    return X, Y


def main():
    ###Preprocessing
    # Imports file as into a dataframe format, removing first row
    data = import_file('Accountability_SQRPratings_2019-2020_SchoolLevel_v20200305_9to12.csv')
    #Separate dataset into independent and dependent sets
    np.set_printoptions(threshold=np.inf)
    data.reset_index()
    print(data.columns.values)
    X, Y = split_variables(data)



    print("X - Missing values/Catagorical values")
    print(X[0])

    # Fill in missing data points
    fill_missing_values(X,2,67)

    print("X - Missing values replaced")
    print(X[0])

    #Encoding categorical values
    need_of_encode = [0, 6, -6]
    X = encode_columns(X,need_of_encode)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(X[:, 0:len(X[0])])
    X[:, 0:len(X[0])]= imputer.transform(X[:, 0:len(X[0])])
    X[:, 0:len(X[0])]= imputer.transform(X[:, 0:len(X[0])])

    print("X - Categorical values replaced")
    r = len(X[0])
    c = len(X)
    for j in range(c):
        for i in range(r):
            if isinstance(X[j,i], str):
                X[j,i] = float(X[j,i])

    print('x')
    print(X[0])
    print(data.iloc[0,:].values)


    #Spliting the sets further into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    print("X Train")
    print(X_train[0])
    print("X Test")
    print(X_test[0])
    print("Y Train")
    print(Y_train)
    print("Y Test")
    print(Y_test)

    #Feature scaling
    sc = StandardScaler()
    X_train[:,16:] = sc.fit_transform(X_train[:,16:])
    X_test[:,16:] = sc.transform(X_test[:,16:])
    print('X scaled')
    print(X_test[0])
    print(X_train[:, 74])

    #Training simple linear regression model
    #parameters include training sets for both independent and dependent variables
    #Xtrain attendance
    ind = X_train[:,74]
    print(ind)
    print(len(ind))
    print(Y_train)
    print(len(Y_train))
    ind = ind.reshape(-1,1)
    Y_train = Y_train.reshape(-1,1)
    regressor = LinearRegression()
    regressor.fit(ind, Y_train)

    y_pred = regressor.predict((X_test[:,74]).reshape(-1,1))


    #Visualize results
    plt.scatter(ind,Y_train, color='red')
    plt.plot(ind, regressor.predict(ind),color = 'blue')
    plt.title('CPS High Schools Quality Scores vs Determinants')
    plt.xlabel('Student Performance')
    plt.ylabel('Overall School Quality Score')
    plt.show()


if __name__ == '__main__':
    main()