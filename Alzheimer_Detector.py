from test import Alzheimer_Detector
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Input information of the patient')

parser.add_argument(
                    '-i', '--info',
                     type=str,
                     required=True,
                     default= '0,0,0,0,0,0,0,0'
                    )
args = parser.parse_args()

if __name__ == '__main__':

    patient_input = args.info.split(',')
    model = Alzheimer_Detector()
    predict = model.user_predict(patient_input)

    print('Predection: ', predict)