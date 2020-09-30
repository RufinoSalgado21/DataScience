import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def read_file(filename):
    path = os.environ['PYTHONPATH'] + os.path.sep + 'datasets' + os.path.sep + filename
    dataset = pd.read_csv(path, skiprows=[0])
    print('Dataset read.')
    return dataset

def replace_NaN(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X)
    X = imputer.transform(X)
    return X

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
    pd.set_option('display.max_rows', None)
    path = "misconduct_report_eoy2019_districtlevel-1.csv"
    dataset = read_file(path)
    #print(dataset)
    print(dataset.columns)
    newdf = dataset.drop(dataset.index[-51:])
    print(newdf)


if __name__ == '__main__':
    main()

df = pd.DataFrame({'id':['a','a','b','c','c'], 'words':['asd','rtr','s','rrtttt','dsfd'], 'num':[1,2,2,3,4]})
print(df)

j = df.groupby('id')['words'].apply(','.join)
df['num'] = df['num'].apply(str)
n = df.groupby('id')['words','num'].agg(','.join)
print(n)
print(df)
print(j)
