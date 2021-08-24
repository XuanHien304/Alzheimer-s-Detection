# Alzheimer's Detection 
data source: https://www.oasis-brains.org/

The project is to detect Alzheimer's disease using Machine Learning tool on MRI dataset.

We use 5 model of Machine learning (Logistic Regression, AdaBoost, xgboost, Random Forest and Decision Tree) and compare them with each other.

## Installation

Create virtual environment

```bash
conda create -n Alzheimer python=3.8
conda activate Alzheimer
```
Install dependencies:

```bash
pip install -r requirements.txt
```

Download and set up data by running

```bash
bash setup_data.sh
```   

## Usage
Run and save model
```python
python train.py
```
Expected output:

```
----------------Result--------------
model    F1_score    Precision  Recall

``` 
Make sure folder ```models``` (save best model) exists

## Prediction

To make the prediction

```bash
  python Alzheime_Detector.py -i *8-Features
```
Each features seperates by comma( , )
  
## Feature
Make sure your input is in exactly in order

| Feature |   Descripstion     
| :-------- | :-----------------|
| `M/F` | Male of Female
| `Age` | Age of patient 
| `EDUC` | Years of education 
| `SES` | Socioeconomic Status
| `MMSE` | [Mini Mental State Examination](http://www.dementiatoday.com/wp-content/uploads/2012/06/MiniMentalStateExamination.pdf) 
| `eTIV	` | [Estimated Total Intracranial Volume](https://link.springer.com/article/10.1007/s12021-015-9266-5)
| `nWBV` | [Normalize Whole Brain Volume](https://pubmed.ncbi.nlm.nih.gov/11547042/) 
| `ASF` | [Atlas Scaling Factor](https://www.sciencedirect.com/science/article/abs/pii/S1053811904003271) 

The prediction of patient will be Demented or Nondemented

##Make prediction using Streamlit API

```bash
streamlit run streamlit.py
```  
Then go to the local link and enter patient information

Press `Make prediction` button to get the result.