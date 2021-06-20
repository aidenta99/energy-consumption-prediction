import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import feature_engineering

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

DATA_DIR = Path("data/")
IMG_DIR = Path("figures/")

data = pd.read_csv(DATA_DIR / "energy_consumption_data.csv")
data.head()
data.describe()
data.describe().columns

# sns.histplot(data["energy consumption"])
# plt.title("Energy consumption over time")
# plt.savefig(IMG_DIR / "histplot.png")

boxplot = sns.boxplot(data['energy consumption']).get_figure()
boxplot.savefig(IMG_DIR / "boxplot.png")

data.hist(bins=50, figsize=(20,15))
plt.savefig(IMG_DIR/"attribute_histogram_plots")
plt.show()

boxplot = sns.boxplot(x='year', y='energy consumption', data=data).get_figure()
boxplot.savefig(IMG_DIR / "boxplot2.png")

boxplot = sns.boxplot(x=data['dayofweek'], y=data['energy consumption']).get_figure()
boxplot.savefig(IMG_DIR / "boxplot3.png")

corr = data.drop(["is_weekend"], axis=1).corr()
heatmap = sns.heatmap(corr, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

# upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
# highly_correlated = [column for column in upper.columns if any(upper[column] >= 0.90)]
# print(highly_correlated)

high_corr_var=np.where(corr>=0.9)
high_corr_var=[(corr.columns[x],corr.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
print(high_corr_var)
# corr.style.background_gradient(cmap='coolwarm')
