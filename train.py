import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, auc, classification_report
import os
import argparse

from src.utils import f1_score, precision, recall
from src import model
from src.dataset import read_data, target
import warnings
warnings.filterwarnings("ignore")
acc = []
PredictedOutput =[]

def run(model):
    df = read_data('./src/data/oasis_longitudinal.csv')
    X, Y = target(df)

    # splitting into three sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

    # Feature scaling
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Training model
    model_train = model.model.values()
    for models in model_train:
        models.fit(X_train, Y_train)

    # Prediction
        y_pred = models.predic(X_test)
        PredictedOutput.append(y_pred)
        f1, pre, rec = f1_score(Y_test, y_pred), precision(Y_test, y_pred), recall(Y_test, y_pred)

        acc.append([f1, pre, rec])
    result = pd.DataFrame(acc, columns=['F1', 'Precision', 'Recall'])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    
    args = parser.parse_args()

    run(model=args.model)

