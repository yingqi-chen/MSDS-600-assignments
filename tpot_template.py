import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

path = '../../w2/clean_churn_data.csv'


def train_model():
    tpot_data = pd.read_csv(path, index_col='customerID')
    features = tpot_data.drop('Churn', axis=1)
    training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['Churn'], stratify=tpot_data['Churn'], random_state=42)

    best_pipeline = XGBClassifier(learning_rate=0.01, max_depth=6, min_child_weight=14, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0)
    
    best_pipeline.fit(training_features, training_target)

    return best_pipeline

