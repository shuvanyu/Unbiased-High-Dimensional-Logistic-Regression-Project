import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
np.random.seed(1)
torch.manual_seed(1)

# Refinement done
# Combine 1,2,3,4
# Downsampling
# Spread 9, 19, 41

def retrieve_heart_data(sample_ratio):
    # Read in data
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca',
               'thal', 'prediction']
    df = pd.read_csv("data/heart_data.data", header = None, names = columns)
    df = (df.replace('?', np.nan)).dropna()

    # Take fraction of rows
    df = df.sample(frac=sample_ratio).reset_index(drop=True)

    # Set all class labels above 1 to 1
    pred =  df.iloc[:, -1]
    pred.loc[pred > 1] = 1

    # Get covariates
    df = df.iloc[: , :-1]
    
    # Drop categorical columns
    df.drop({'cp','restecg', 'slope'}, inplace=True, axis=1)

    X = torch.from_numpy(df.astype('float64').to_numpy())

    # Shuffle columns
    shuffle_idxs = np.random.permutation(range(0,X.shape[1]))
    X = X[:,shuffle_idxs]

    # Normalize data
    scaler = RobustScaler()
    X = torch.from_numpy(scaler.fit_transform(X.numpy()))

    y = torch.from_numpy(pred.astype('float64').to_numpy())
    
    return X, y