# https://drive.google.com/file/d/1ruLi0vo0PPESxvb-uOa1POhrXfkGsy7F/view?usp=sharing
# Ref: https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb

import requests
import os
import shutil

FILE_ID = '1ruLi0vo0PPESxvb-uOa1POhrXfkGsy7F'
DESTINATION = './data/oasis_longitudinal.csv'

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    os.makedirs('./data')
    download_file_from_google_drive(FILE_ID, DESTINATION)