# -*- encoding = utf-8 -*-
#import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random
import matplotlib.pyplot as plt
import glob
import geopandas as gpd
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from shapely.geometry import Point, Polygon
import pyproj
import joblib
from shapely.ops import transform
#load data
dfr =  pd.read_csv('./data/traindt.csv',index_col=False)
#adjust for real_area ratio
traindt, testdt = train_test_split(dfr,test_size=0.2,shuffle=True,random_state=2)
train_x,train_y = traindt.values[:,:-1],traindt.values[:,-1]
test_x, test_y = testdt.values[:,:-1], testdt.values[:,-1]
train_y = np.log(train_y)
test_y = np.log(test_y)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_x,train_y)
bsregr = rf_random.best_estimator_
joblib.dump(bsregr,'../model/randr_bestmodel.joblib')