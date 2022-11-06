# library import
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

import bentoml

# data import
data = '/home/lisa/Documents/ml-projects/fetal-health-classification/dataset/fetal_health.csv'
df = pd.read_csv(data)

# preprocessing
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('percentage_of_time', 'pot')
categories_log = ['mean_value_of_short_term_variability',
                  'mean_value_of_long_term_variability',
                  'histogram_number_of_peaks',
                  'histogram_variance']
                  
categories_sqrt = ['accelerations',
                   'fetal_movement',
                   'light_decelerations',
                   'severe_decelerations',
                   'prolongued_decelerations',
                   'pot_with_abnormal_long_term_variability',
                   'histogram_number_of_zeroes']
                   
for c in categories_log:
    
    df[c] = np.log1p(df[c])
    
for s in categories_sqrt:
    
    df[s] = np.sqrt(df[s])
    
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_full_train = df_full_train['fetal_health'].values

del df_full_train['fetal_health']

scaler = MinMaxScaler()

scaler.fit(df_full_train)

df_full_train_arr    = scaler.transform(df_full_train)
df_full_train_scaled = pd.DataFrame(df_full_train_arr, columns=df_full_train.columns)

df_full_train = df_full_train_scaled.reset_index(drop=True)

dv = DictVectorizer(sparse=False)

# let's create the feature matrix for df_full_train
full_train_dict = df_full_train.to_dict(orient = 'records')
X_full_train    = dv.transform(full_train_dict)

# let's train the final model
RF = RandomForestClassifier(n_estimators = 110,
                            max_depth = 15,
                            min_samples_leaf = 1,
                            random_state=1)

RF.fit(X_full_train, y_full_train)

# let's save it with BentoML
bentoml.sklearn.save_model("fethal_health_model", RF, custom_objects={"dictVectorizer": dv})
