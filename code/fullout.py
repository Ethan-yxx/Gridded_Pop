# -*- encoding = utf-8 -*-
import joblib
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shapely.geometry import Point, Polygon
from sklearn.metrics import mean_squared_error, r2_score
import subprocess
regr = joblib.load('../model/randr_bestmodel.joblib')

fulldt = pd.read_csv('../data/data36.csv',index_col=False)

fulldtx = fulldt.drop(['id','x','y','rea_area'],axis=1)
fulldt['preds'] = regr.predict(fulldtx)
fulldt['preds'] = np.exp(fulldt['preds'])
fulldt['preds'] = fulldt['preds']*fulldt['rea_area']

cntwgs = gpd.read_file('../arcdata/cnt4.shp',index=False).rename({'pop':'popqipu'},axis=1)
fulldt_geo = fulldt.apply(lambda x: Point(x['x'],x['y']),axis=1)
fulldtg = gpd.GeoDataFrame(fulldt,geometry = fulldt_geo)
fulldtg.crs = {'init':'epsg:4326'}
fulldtsj = gpd.sjoin(fulldtg,cntwgs, how='left',op= 'within').dropna()
fulldt2 = fulldtsj.drop(['geometry','index_right','cnt','city','prov','popqipu'],axis=1)

fulldtsjag = fulldtsj.groupby(['prov','city','cnt','popqipu']).agg({'preds':'sum'}).reset_index()
fulldtsjag['ratio'] = fulldtsjag['preds']/fulldtsjag['popqipu']

fulldtsj = fulldtsj.merge(fulldtsjag.drop('preds',axis=1),on = ['prov','city','cnt','popqipu'], how = 'left')
fulldtsj['preds'] = fulldtsj['preds']/fulldtsj['ratio']
fulldtsjo1  = fulldtsj[['id','x','y','preds']].copy()
fulldtsjo1['x'] = np.round(fulldtsjo1['x']*1000)/1000
fulldtsjo1['y'] = np.round(fulldtsjo1['y']*1000)/1000

fulldtsjo1.to_csv('../data/preds.csv',index=False)