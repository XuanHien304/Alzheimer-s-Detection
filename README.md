# Alzheimer's Detection 
dataset source: https://www.oasis-brains.org/

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