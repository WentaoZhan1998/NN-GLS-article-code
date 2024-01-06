import netCDF4
from netCDF4 import num2date
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")

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

df = pd.read_csv('covariate0618.csv').iloc[:,1:9]
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
df.to_csv('covariate0618.csv')





