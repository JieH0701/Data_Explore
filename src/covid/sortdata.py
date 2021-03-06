import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import dates as dt

import matplot_date.CreateMatplotXdata as matplt


for dirname, _, filenames in os.walk('/Users/hou0002j/Projekt/covid-19/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        file_path = os.path.join(dirname, filename)
        df_covid_19 = pd.read_csv(os.path.join(dirname, filename))

df_covid_19['ObservationDate'] = pd.to_datetime(df_covid_19['Date'])
df_covid_19['Outbreak'] = 'COVID_2019'

df_covid_19.loc[df_covid_19['Country/Region'] == 'US', 'Country/Region'] = 'United States'
df_covid_19.loc[df_covid_19['Country/Region'] == 'Mainland China', 'Country/Region'] = 'China'
df_covid_19.loc[df_covid_19['Country/Region'] == 'Viet Nam', 'Country/Region'] = 'Vietnam'
df_covid_19.loc[df_covid_19['Country/Region'] == 'UK', 'Country/Region'] = 'United Kingdom'
df_covid_19.loc[df_covid_19['Country/Region'] == 'South Korea', 'Country/Region'] = 'Korea, South'
df_covid_19.loc[df_covid_19['Country/Region'] == 'Taiwan, China', 'Country/Region'] = 'Taiwan'
df_covid_19.loc[df_covid_19['Country/Region'] == 'Hong Kong SAR, China', 'Country/Region'] = 'Hong Kong'
df_covid_19.loc[df_covid_19['Country/Region'] == 'Germany', 'Country/Region'] = 'Germany'
df_covid_19.loc[df_covid_19['Country/Region'] == 'Italy', 'Country/Region'] = 'Italy'

data_italy = df_covid_19.loc[df_covid_19['Country/Region'] == 'Italy']
data_germany = df_covid_19.loc[df_covid_19['Country/Region'] == 'Germany']
data_usa = df_covid_19.loc[df_covid_19['Country/Region'] == 'United States']
data_china = df_covid_19.loc[df_covid_19['Country/Region'] == 'china']
data_korea_south = df_covid_19.loc[df_covid_19['Country/Region'] == 'Korea, South']

ax = matplt.CreateMatplotXdata(data_italy['ObservationDate']).create_matplot_x_data()
ax.plot(data_italy['ObservationDate'], data_italy['Confirmed'], 'r--',
        data_germany['ObservationDate'] + np.timedelta64(-7, 'D'),
        data_germany['Confirmed'], 'b-')

usa_data_corr = data_usa.loc[data_usa['Confirmed'] > 50]
ax_usa = matplt.CreateMatplotXdata(usa_data_corr['ObservationDate']).create_matplot_x_data()

ax_usa.plot(usa_data_corr['ObservationDate'], usa_data_corr['Confirmed'], 'y-')
plt.show()

X_mask_cat = ['Confirmed', 'Region_enc', 'Month', 'Week']
train_test_italy = data_italy.copy()