import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))

def read_data(filename):
    path = os.environ[
               "PYTHONPATH"] + os.path.sep + "datasets" + os.path.sep + filename
    dataset = pd.read_csv(path)
    print("Dataset read.")
    #print(dataset.head())
    return dataset

def main():
    np.set_printoptions(threshold=np.inf)
    pd.set_option('display.max_columns', 10)
    file = "SQRPratings_2019-2020_SchoolLevel_9to12_edited.csv"
    dataset = read_data(file)
    cols = dataset.columns.values
    indices = [0, 1, 2, 3, 4, 5, 38, 39, 40, 41]
    cols = np.delete(cols, indices)
    X = dataset.iloc[:,6:].values
    Y = dataset.iloc[:,3].values

    #Encode X feature at index 65 containing catagorical values
    print("Encoding catagorical values in X")
    col_needs_encoding = 65
    transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [col_needs_encoding])], remainder="passthrough")
    X = np.array(transformer.fit_transform(X[:,:]))

    '''
    0Moderately
    1Not Enough Data
    2Not Yet
    3Organized
    4Partially Organized
    5Well
    '''
    #Formatting column headers array for easier navigation
    #Encoding removed index 61 from X and inserted 6 new indices at the beginning for each catagorical value it contained
    txt = 'My Voice, My School 5 Essentials Survey  Score'
    indices = [61]
    new_indices = [txt+' Moderately',txt+' Not Enough Data',txt+' Not Yet',txt+' Organized',txt+' Partially',txt+' Well']
    cols = np.delete(cols,indices)
    cols = np.insert(cols, 0, new_indices)

    #Replace NaN values
    print("Replacing missing values with means in X")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:,:])
    #No SAT data means 4 cols will be removed changing the shape of the array
    X = imputer.transform(X)

    #Generate train and test sets for X and Y
    print("Splitting sets into training and testing sets")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    #Displaying columns headers with associated indices for navigation
    print("Dataset independent variable indices")
    for i in range(len(cols)):
        print(str(i) + '. ' + cols[i])

    #Adjusting the index of the preprocessed X dataset will change the independent variable to compare against Y
    index = 55
    print('Independent variable: ' + cols[index])
    independent_train = X_train[:,index].reshape(-1,1)
    independent_test = X_test[:,index].reshape(-1,1)
    regressor = LinearRegression()
    regressor.fit(independent_train, Y_train)

    Y_predict = regressor.predict(independent_test)

    #finally, plot the correlation
    plt.scatter(independent_train, Y_train, color='red')
    plt.plot(independent_train,regressor.predict(independent_train), color='blue')
    plt.title('CPS SQRP Total Points Earned vs ' + cols[index])
    plt.xlabel(cols[index])
    plt.ylabel('SQRP Total Points')
    plt.show()


if __name__ == '__main__':
    main()
