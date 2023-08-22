import netCDF4
from netCDF4 import num2date
import numpy as np
import os
import pandas as pd
#from geopy.geocoders import Nominatim
#geolocator = Nominatim(user_agent="geoapiExercises")

df0 = pd.read_csv('covariate0605.csv')


file_location = './/air.2m.2022.nc'
f = netCDF4.Dataset(file_location)
air = f.variables['air']

latitudes = f.variables['lat'][:]
longitudes = f.variables['lon'][:]

time_dim, lat_dim, lon_dim = air.get_dims()
time_var = f.variables[time_dim.name]
times = num2date(time_var[:], time_var.units)

file_location = './/rhum.2m.2022.nc'
f = netCDF4.Dataset(file_location)
rhum = f.variables['rhum']

file_location = './/acpcp.2022.nc'
f = netCDF4.Dataset(file_location)
acpcp = f.variables['acpcp']

file_location = './/pres.sfc.2022.nc'
f = netCDF4.Dataset(file_location)
pres = f.variables['pres']

file_location = './/uwnd.10m.2022.nc'
f = netCDF4.Dataset(file_location)
uwnd = f.variables['uwnd']

file_location = './/vwnd.10m.2022.nc'
f = netCDF4.Dataset(file_location)
vwnd = f.variables['vwnd']

from datetime import datetime
for i, t in enumerate(times):
    if datetime.fromisoformat(t.isoformat()) == datetime.fromisoformat('2022-06-18'):
        filename = 'covariate0618.csv'
        df = pd.DataFrame({'long':np.array(longitudes).reshape(-1), 'lat':np.array(latitudes).reshape(-1),
                          'prec':np.array(acpcp[i, :, :]).reshape(-1), 'temp':np.array(air[i, :, :]).reshape(-1),
                           'pres':np.array(pres[i, :, :]).reshape(-1), 'rh':np.array(rhum[i, :, :]).reshape(-1),
                          'uwnd':np.array(uwnd[i, :, :]).reshape(-1), 'vwnd':np.array(vwnd[i, :, :]).reshape(-1)
                           })
        df.to_csv(filename)

df = df.loc[(df['long']>=np.min(df0['long'])) &
       (df['long']<=np.max(df0['long'])) &
       (df['lat']>=np.min(df0['lat'])) &
       (df['lat']<=np.max(df0['lat']))
]
df.to_csv(filename)

df = pd.read_csv('pm25_2022.csv').iloc[:,1:5]
subdf = df.loc[df['Date']=='06/18/2022']
subdf.to_csv('pm25_0618.csv')

df = pd.read_csv('covariate0628.csv').iloc[:,1:9]
df0 = pd.read_csv('covariate0605.csv').iloc[:,1:9]
mask = np.zeros(df.shape[0], dtype=bool)
for i in range(df.shape[0]):
    print(i)
    location = geolocator.geocode(str(df['lat'][i]) + "," + str(df['long'][i]))
    try:
        l = location[0].split()
        if l[len(l)-1]=='States':
            mask[i] = True
            print(str(i) + ' in U.S.')
    except Exception:
        print(str(i) + ' missed location')
        pass

df = df.loc[mask]
df.to_csv('covariate0628_filtered.csv')

import numpy as np
import os
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
    
np.random.seed(2) # set seed (needed for reproducible results
N = 2000
rndn_sample = pd.DataFrame({'x':np.random.uniform(x_min,x_max,N),'y':np.random.uniform(y_min,y_max,N)}) # actual generation
# re-save results in a geodataframe
rndn_sample = gpd.GeoDataFrame(rndn_sample, geometry = gpd.points_from_xy(x=rndn_sample.x, y=rndn_sample.y),crs = us.crs)
inUS = rndn_sample['geometry'].apply(lambda s: s.within(us.geometry.unary_union)) # check if within the U.S. bounds
rndn_sample.loc[inUS,:].plot()
plt.savefig(".//temp_figure//US.png")

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

df1 = pd.read_csv('covariate0618.csv')
#df2 = pd.read_csv('pm25_0628.csv')
df2 = pd.read_csv('pm25_0618.csv')
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

#fig1, ax1 = p.figure(1)
#plot = ax1.imshow(z,interpolation='none',cmap=p.cm.jet,origin='lower')  
#fig1.colorbar(plot, ax=ax1)
#ax1.set_xlabel('Longitude')
#ax1.set_ylabel('Latitude')
#ax1.set_title('title')

plt.clf()
fig, ax = plt.subplots(figsize=(9, 5))
c = ax.scatter(x = x_test, y = y_test,s = 10, c = z_test, marker = 's', alpha = 0.7)
ax.plot(np.array(df2['Longitude']), np.array(df2['Latitude']), 'ro', c = 'orange', markersize = 4)
ax.set_title('')
fig.colorbar(c, ax=ax)
plt.savefig(".//temp_figure//US_heat_0618.png")

plt.clf()
fig, ax = plt.subplots()
c = ax.scatter(x = lon[idx_new], y = lat[idx_new], c = z)
ax.set_title('pcolormesh')
fig.colorbar(c, ax=ax)
plt.savefig(".//temp_figure//US_orig.png")





