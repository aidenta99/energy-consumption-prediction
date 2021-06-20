import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import numpy as np


data = pd.read_csv(DATA_DIR / "energy_consumption_data.csv")
data.head()
data.describe()
data.describe().columns

data.dropna(subset=['Sample Time 1'], inplace = True)

data['Sample Time 1'] = pd.to_datetime(data['Sample Time 1'])
data['N'] = data['N'].astype(str)

# features['Yr Sold'] = features['Yr Sold'].astype(str)

data['year'] = data['Sample Time 1'].dt.year
data['year'] = data['year'].astype(str)
data['month'] = data['Sample Time 1'].dt.month
data['month'] = data['month'].astype(str)
data['dayofweek'] = data['Sample Time 1'].dt.dayofweek
data['dayofweek'] = data['dayofweek'].astype(str)
data['is_weekend'] = np.where(data['dayofweek'].isin([5, 6]),1,0)
data['is_open'] = np.where((data['Sample Time 1'].dt.hour > 8) \
                            & (data['Sample Time 1'].dt.hour < 23),1,0)

# Sample Time 1: 3 (remove those rows, see the code below)
# TIN: 3
# TRE: 34 (replace by average values)
# MRE: 34 (replace by average values)
# MME: 105
# N: 36
# energy consumption: 44

# impute nulls for continuous data
data.TRE = data.TRE.fillna(data.TRE.median())
data.TIN = data.TIN.fillna(data.TRE.median())
data.MRE = data.MRE.fillna(data.MRE.median())
data.MME = data.MME.fillna(data.MME.median())
data.N = data.N.fillna(data.N.mode()[0])
data['energy consumption'] = data['energy consumption'].fillna(data['energy consumption'].median())

data.head()