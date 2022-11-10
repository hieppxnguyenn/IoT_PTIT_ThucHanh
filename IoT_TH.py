import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(filepath_or_buffer="/Users/linh/Downloads/housing.csv", delim_whitespace=True, names=name)
df.head()

data = df.values[:, 13]

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['MEDV'], axis=1),
    df['MEDV'],
    test_size=0.2,
    random_state=1)
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred_1 = lm.predict(X_test)
print(r2_score(y_test, y_pred_1))
corr = df.corr()

plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True, annot_kws={'size': 12}, cmap='coolwarm')
plt.show()

x = df.drop(labels=['MEDV'], axis=1)
# x = df.drop(labels=['MEDV','CHAS'], axis=1)
# x = df.drop(labels=['MEDV','CHAS','DIS'], axis=1)
# # x = df.drop(labels=['MEDV','CHAS','DIS', 'B'], axis=1)
y = df['MEDV']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
lm = LinearRegression()
lm.fit(train_x, train_y)
predicted_y = lm.predict(test_x)
res = pd.DataFrame({'Actual': test_y, 'Predict': predicted_y})
print(res)
print('coef:', lm.coef_)
print('intercept:', lm.intercept_)
r2 = r2_score(test_y, predicted_y)
mse = mean_squared_error(test_y, predicted_y)
print('mse: ', mse, 'r2: ', r2)






