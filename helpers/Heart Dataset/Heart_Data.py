import pandas as pd
import numpy as np

# Refinement done
# Combine 1,2,3,4
# Downsampling
# Spread 9, 19, 41

def data_extract_and_refine():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca',
               'thal', 'prediction']
    df = pd.read_csv("heart_data.data", header = None, names = columns)
    df = (df.replace('?', np.nan)).dropna()

    pred =  df.iloc[:, 13:14]
    df = df.iloc[: , :-1]
    pred.loc[(pred.prediction > 1), 'prediction'] = 1
    
    y1 = pd.get_dummies(df.cp, prefix='cp')
    y2 = pd.get_dummies(df.restecg, prefix='restecg')
    y3 = pd.get_dummies(df.slope, prefix='slope')
    y = pd.concat([y1, y2, y3], axis=1)
    df.drop({'cp','restecg', 'slope'}, inplace=True, axis=1)
    df = pd.concat([df, y, pred], axis=1)
    
    df_sampled = df.sample(frac=0.58)
    
    return df_sampled

df = data_extract_and_refine()