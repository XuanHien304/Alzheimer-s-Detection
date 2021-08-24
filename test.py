import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler 

class Alzheimer_Detector():

    def __init__(self):
        best_model = os.listdir('./models')
        self.model = joblib.load('./models' + '/' + str(best_model[0]))

    def user_predict(self, info):
        '''
            Input: feature of the patient

            Output: Alzheimer's Demented or Not

        '''
        info_np = np.array(info,float).reshape(1,-1)
        scaler = MinMaxScaler().fit(info_np)
        info_np = scaler.transform(info_np)
        prediction = self.model.predict(info_np)
        
        if prediction[0] == 0:
            output = "Nondemented"
        else:
            output = "Demented" 
        return output 



