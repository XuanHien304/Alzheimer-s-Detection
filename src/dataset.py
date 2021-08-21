import pandas as pd
import numpy as np


def read_data(data_dir):
    '''
    Input:
        data_dir <str> : path to data file (csv), can be train set or test set
    Returns:
        Dataframe after clean

    '''
    df = pd.read_csv(data_dir)
    ind = df[df.Group=='Converted'].index
    df = df.drop(ind,axis=0)
    df.reset_index(drop=True,inplace=True)
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    df['Group'] = df['Group'].replace(['Nondemented','Demented'], [0,1])
    df['M/F'] = df['M/F'].replace(['M','F'], [1,0])
    df = df.drop(['MRI ID', 'Visit', 'Hand', 'MR Delay'], axis=1)
    # Fillna NaN
    df['SES'].fillna(df.groupby('EDUC')['SES'].transform('median'),inplace = True)
    df['MMSE'].fillna(df['MMSE'].mean(),inplace = True)
    
    return df

def target(df):
    X = df[['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV','ASF']]
    Y = df['Group'].values

    return X, Y

     