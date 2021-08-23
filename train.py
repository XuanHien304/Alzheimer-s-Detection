import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, auc, classification_report
import os
import argparse
import joblib
import shutil

from src.utils import f1_score, precision, recall
from src import model
from src.dataset import read_data, target
import warnings
warnings.filterwarnings("ignore")
acc = []
PredictedOutput =[]


def run(model):
    best_f1  = 0
    df = read_data('./src/data/oasis_longitudinal.csv')
    X, Y = target(df)

    # splitting into three sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

    # Feature scaling
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Training model
    model_train = model.model
    for name, models in model_train.items():
        models.fit(X_train, Y_train)

    # Prediction
        y_pred = models.predict(X_test)
        PredictedOutput.append(y_pred)
        f1, pre, rec = f1_score(Y_test, y_pred), precision(Y_test, y_pred), recall(Y_test, y_pred)

        acc.append([name, f1, pre, rec])
        if f1 > best_f1:
            best_f1 = f1
            best_model = model_train[name]
    # Save model
    joblib.dump(
        best_model,
        os.path.join('./models', str(models)[:4] + "model.pkl")
                )


    result = pd.DataFrame(acc, columns=['Model', 'F1', 'Precision', 'Recall'])
    print('------------------------Result--------------------------')
    print()
    print(result)

   

if __name__ == "__main__":
    if os.path.exists('./models'):
        shutil.rmtree('./models')
    os.makedirs('./models')
    run(model=model)
    
    
