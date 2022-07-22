# -*- encoding = utf-8 -*-
import joblib
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shapely.geometry import Point, Polygon
from sklearn.metrics import mean_squared_error, r2_score

#
preds = pd.read_csv('../data/preds.csv',index_col=False)
wp = pd.read_csv('../otherproduct/worldpop100_sel.csv',index_col=False)
gpw = pd.read_csv('../otherproduct/gpw_sel.csv',index_col=False)
#load bd pop
#load cntwgs data county level 
townwgs = gpd.read_file('../arcdata/town.shp')
townqipu = pd.read_csv('../data/townqipuhz.csv',index_col=False,sep=',',header=None)
townqipu.columns = ['cnt','town','popqipu']
#select towns whose boundaries are not changes in recent years
cntlist = ['建德市','淳安县','淳安县','滨江区','西湖区']
townqipu = townqipu[townqipu['cnt'].isin(cntlist)].copy()
print (townqipu.shape)
townwgs = townwgs.merge(townqipu,on = ['cnt','town'],how = 'left').dropna()
towndis = townwgs.dissolve(by = 'city')
[xmin,ymin, xmax,ymax] = towndis['geometry'].bounds.values.tolist()[0]
#select grid cells within the counties selected
preds = preds[(preds['x']>=xmin )& (preds['x']<=xmax )& (preds['y']>=ymin) & (preds['y']<=ymax)].copy()
wp = wp[(wp['x']>=xmin) & (wp['x']<=xmax) & (wp['y']>=ymin ) & (wp['y']<=ymax)].copy()
gpw = gpw[(gpw['x']>=xmin )& (gpw['x']<=xmax )& (gpw['y']>=ymin) & (gpw['y']<=ymax)].copy()


fulldt_geo = preds.apply(lambda x: Point(x['x'],x['y']),axis=1)
fulldtg = gpd.GeoDataFrame(preds,geometry = fulldt_geo)
fulldtsj = gpd.sjoin(fulldtg,townwgs, how='left',op= 'within').dropna()
predsag = fulldtsj.groupby(['cnt','town','popqipu']).agg({'preds':'sum'}).reset_index()
#load worldpopdata and gpw data
wp_geo = wp.apply(lambda x: Point(x['x'],x['y']),axis=1)
gpw_geo = gpw.apply(lambda x: Point(x['x'],x['y']),axis=1)
#
wptg = gpd.GeoDataFrame(wp,geometry = wp_geo)
gpwtg = gpd.GeoDataFrame(gpw,geometry = gpw_geo)

wpsj = pd.DataFrame(gpd.sjoin(wptg,townwgs, how='left',op= 'within').dropna())
wpsjag = wpsj.groupby(['cnt','town','popqipu']).agg({'data':'sum'}).reset_index().rename({'data':'wp'},axis=1)
gpwsj = pd.DataFrame(gpd.sjoin(gpwtg,townwgs, how='left',op= 'within').dropna())
gpwsjag = gpwsj.groupby(['cnt','town','popqipu']).agg({'data':'sum'}).reset_index().rename({'data':'gpw'},axis=1)

odt = predsag.merge(wpsjag,on = ['cnt','town','popqipu'],how='left').dropna()
odt = odt.merge(gpwsjag,on= ['cnt','town','popqipu'],how='left').dropna()

odt.to_csv('../data/town_predsnn.csv',index= False)

cnt = odt
city = cnt.groupby(['cnt'])['popqipu','preds','wp','gpw'].sum().reset_index()
city['predsratio'] = city['preds']/city['popqipu']
city['wpratio'] = city['wp']/city['popqipu']
city['gpwratio'] = city['gpw']/city['popqipu']
cnt = cnt.merge(city[['cnt','predsratio','wpratio','gpwratio']],on = ['cnt'], how= 'left')
cnt['preds2'] = cnt['preds']/cnt['predsratio']
cnt['wp2'] = cnt['wp']/cnt['wpratio']
cnt['gpw2'] = cnt['gpw']/cnt['gpwratio']

r2 = [r2_score(cnt['popqipu'],cnt['preds2']),
        r2_score(cnt['popqipu'],cnt['wp']),r2_score(cnt['popqipu'],cnt['gpw'])]
rrmse =  [np.sqrt(mean_squared_error(cnt['popqipu'],cnt['preds2'])),
            np.sqrt(mean_squared_error(cnt['popqipu'],cnt['wp2'])),
            np.sqrt(mean_squared_error(cnt['popqipu'],cnt['gpw2']))]

result = [r2,rrmse]
resultdf = pd.DataFrame(result,columns=['ours','wp','gpw'])
resultdf.to_csv('../data/town_resultnn.csv',index=False)