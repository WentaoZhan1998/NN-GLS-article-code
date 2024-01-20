# This file generate the heatmap on US in Figure 3(a), S28.

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from shapely.geometry import Point
from scipy import interpolate
df0 = pd.read_csv('covariate0605.csv')
url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_nation_20m.zip"
us = gpd.read_file(url).explode()
us = us.loc[us.geometry.apply(lambda x: x.exterior.bounds[2])<-60]
x_min,y_min,x_max,y_max = np.array([np.min(df0['long']), np.min(df0['lat']),
    np.max(df0['long']), np.max(df0['lat'])])

np.random.seed(2)
N = 100
arr1 = np.mgrid[x_min:x_max:101j, y_min:y_max:101j]

# extract the x and y coordinates as flat arrays
arr1x = np.ravel(arr1[0])
arr1y = np.ravel(arr1[1])

# using the X and Y columns, build a dataframe, then the geodataframe
df = pd.DataFrame({'X':arr1x, 'Y':arr1y})
df['coords'] = list(zip(df['X'], df['Y']))
df['coords'] = df['coords'].apply(Point)
gdf1 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.X, y=df.Y),crs = us.crs)
inUS = gdf1['geometry'].apply(lambda s: s.within(us.geometry.unary_union)) # check if within the U.S. bounds
gdf1.loc[inUS,:].plot()
plt.savefig(".//temp_figure//US.png")

for name in ['0605', '0618', '0704']: #### "0605", "0618" "0704" corresponding to 2019.06.05, 2022.06.18 and 2022.07.04
    df1 = pd.read_csv('covariate'+ name +'.csv')
    df2 = pd.read_csv('pm25_' + name + '.csv')
    df2 = df2.loc[df2.Latitude < 50]
    covariates = df1.values[:,3:]
    aqs_lonlat=df2.values[:,[1,2]]

    from scipy import spatial
    near = df1.values[:,[1,2]]
    tree = spatial.KDTree(list(zip(near[:,0].ravel(), near[:,1].ravel())))
    tree.data
    idx = tree.query(aqs_lonlat)[1]
    df2_new = df2.assign(neighbor = idx)
    df_pm25 = df2_new.groupby('neighbor')['PM25'].mean()
    df_pm25_class = pd.cut(df_pm25,bins=[0,12.1,35.5],labels=["0","1"])
    idx_new = df_pm25.index.values
    pm25 = df_pm25.values
    z = pm25[:,None]

    lon = df1.values[:,1]
    lat = df1.values[:,2]
    normalized_lon = (lon-min(lon))/(max(lon)-min(lon))
    normalized_lat = (lat-min(lat))/(max(lat)-min(lat))
    s_obs = np.vstack((normalized_lon[idx_new],normalized_lat[idx_new])).T

    f = interpolate.Rbf(lon[idx_new], lat[idx_new], z, function = 'inverse')
    x_test = gdf1.loc[inUS,:].X
    y_test = gdf1.loc[inUS,:].Y
    z_test = f(x_test, y_test)

    plt.clf()
    fig, ax = plt.subplots(figsize=(9, 5))
    c = ax.scatter(x = x_test, y = y_test, s = 10, c = z_test, marker = 's', alpha = 0.7)
    ax.plot(np.array(df2['Longitude']), np.array(df2['Latitude']), 'ro', c = 'orange', markersize = 4)
    ax.set_title('')
    fig.colorbar(c, ax=ax)
    plt.savefig(".//temp_figure//US_heat_" + name + ".png")






