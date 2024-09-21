import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# Data Gathering
df2014 = pd.read_csv('/Users/yaakovtouati/Downloads/Fish_2014.csv')
df2015 = pd.read_csv('/Users/yaakovtouati/Downloads/Fish_2015.csv')
df2016 = pd.read_csv('/Users/yaakovtouati/Downloads/Fish_2016.csv')

df = pd.concat([df2014, df2015, df2016], axis=0).reset_index(drop=True)


# Data Preperation
stat = df.describe()
df = df.replace(0, np.nan)
nulls = df.isna().sum()

df.fillna({
    'Weight': df['Weight'].mean(),
    'Length1': df['Length1'].mean(),
    'Length2': df['Length2'].mean(),
    'Length3': df['Length3'].mean(),
    'Height': df['Height'].mean(),
    'Width': df['Width'].mean()
}, inplace=True)

nulls = df.isna().sum()


# Get correlation
cors = df.corr()

df = df.drop('Length2', axis=1)
cors = df.corr()

df = df.drop('Length3', axis=1)
cors = df.corr()

df = df.drop('Width', axis=1)
cors = df.corr()

X = df.drop('Weight', axis=1).copy().values
y = df['Weight'].copy().values


# Split the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the data (we don't want to normalize y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lin_model = LinearRegression()
X = sc.fit_transform(X)

# regressor = LinearRegression()

kfold =KFold(n_splits=10, random_state=1, shuffle=True)

model_kfold_linear = LinearRegression()
results_kfold = cross_val_score(model_kfold_linear, X, y, cv=kfold, scoring='r2')
mean_score = results_kfold.mean()
r2_linear = mean_score

model_kfold_dlecision = DecisionTreeRegressor()
results_kfold = cross_val_score(model_kfold_dlecision, X, y, cv=kfold, scoring='r2')
mean_score = results_kfold.mean()
# r2_linear = mean_score


max_r2 = 0
for i in range(1,10):
    poly_features = PolynomialFeatures(degree=i)
    X_train_poly = poly_features.fit_transform(X_train)
    
    poly_mod = LinearRegression()
    poly_mod.fit(X_train_poly, y_train)
    

    y_train_pred = poly_mod.predict(X_train_poly)
    y_test_pred = poly_mod.predict(poly_features.fit_transform(X_test))
    
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    
    print("Degree : " + str(i))
    print("RMSE Test : " + str(rmse_test))
    print("RMSE Train : " + str(rmse_train))
    print("r2_test : " + str(r2_score(y_test, y_test_pred)))
    print("r2_train : " + str(r2_score(y_train, y_train_pred)))
    
    print("")
    

parameters = {"splitter" : ["best", "random"],
              "max_depth" : [1, 3, 5, 7],
              "min_samples_leaf" : [1,2,3,4,5,6],
              "min_weight_fraction_leaf" : [0.1, 0.2, 0.3, 0.4],
              "max_features" : ["auto", "log2", "sqrt"],
              "max_leaf_nodes" : [None, 10, 20, 30, 40]
    }

tuning_model = GridSearchCV(model_kfold_dlecision, param_grid=parameters, cv=3, scoring='r2')
tuning_model.fit(X,y)

best_params = tuning_model.best_params_
best_score = tuning_model.best_score_


poly_features = PolynomialFeatures(degree=5)
X_train_poly = poly_features.fit_transform(X_train)
poly_mod = LinearRegression()
poly_mod.fit(X_train_poly, y_train)


data = [[22, 11.44]]
data = sc.transform(data)

data_to_predict = poly_features.fit_transform(data)

result = poly_mod.predict(data_to_predict)






























