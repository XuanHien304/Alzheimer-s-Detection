from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

model = {
    'Logistic Regression': LogisticRegression(
        C=100
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5
    ),
    'Adaboost': AdaBoostClassifier(
    n_estimators=4, learning_rate=1, random_state=42
    ),
    'RandomForest': RandomForestClassifier(
    n_estimators=10, max_features=1,
    max_depth=6, random_state=42
    ),
    'XGBoost': XGBClassifier(
        max_depth=10, learning_rate=0.5,gamma=0.4, random_state=42,
        objective = 'binary:logistic', eval_metric = 'logloss' 
    )
}
