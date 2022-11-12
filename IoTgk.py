# Linear Regression
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

weather_df = pd.read_csv("/Users/linh/Downloads/weatherHistory.csv", low_memory=False)
# 96454 rows x 12 columns

weather_df.loc[weather_df['Precip Type'].isnull(), 'Precip Type'] = 'rain'
# get rid of null cells in precip type

weather_corr = weather_df[list(weather_df.dtypes[weather_df.dtypes != 'object'].index)].corr()
sns.heatmap(weather_corr, annot=True)
plt.show()
# correlation

weather_df.loc[weather_df['Precip Type'] == 'rain', 'Precip Type'] = 1
weather_df.loc[weather_df['Precip Type'] == 'snow', 'Precip Type'] = 0
# change data from rain/snow to 1/0

weather_df_num = weather_df[list(weather_df.dtypes[weather_df.dtypes != 'object'].index)]

weather_y = weather_df_num.pop('Temperature (C)')
weather_X = weather_df_num
train_X, test_X, train_y, test_y = train_test_split(weather_X, weather_y, test_size=0.2, random_state=4)
model = LinearRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
np.mean((prediction - test_y) ** 2)
res = pd.DataFrame({'actual': test_y,
                    'prediction': prediction,
                    'diff': (test_y - prediction)})
print(res)
print("coef: ", model.coef_)
print("intercept: ", model.intercept_)
r2 = r2_score(test_y, prediction)
mse = mean_squared_error(test_y, prediction)
print('mse: ', mse, 'r2: ', r2)

