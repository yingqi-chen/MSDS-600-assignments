import pandas as pd
from tpot_template import train_model

path = '../new_churn_data.csv'

def load_data_for_prediction():
    df_need_to_predict =  pd.read_csv(path, index_col='customerID')
    
    # this feature is extra and will cause errors if I want to predict the data with the current model
    new_features = df_need_to_predict.drop('charge_per_tenure', axis=1)

    return new_features

def make_predictions():
    pipeline = train_model()
    features = load_data_for_prediction()
    print(pipeline)
    return pipeline.predict(features)
     

if __name__ == "__main__":
    print('predictions:')
    print(make_predictions())