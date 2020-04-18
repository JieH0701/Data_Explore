import numpy as np  # linear algebra
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.axis
import matplotlib
import matplotlib.pyplot as plt


df_germany = pd.read_csv('./dataset/RKI_COVID19.csv')
df_germany['RegistrationDate'] = pd.to_datetime(df_germany['Meldedatum'])

# get the column name and the quality and completeness of the data
df_germany.info()

bundeslands = df_germany['Bundesland'].unique()
# 'Schleswig-Holstein', 'Niedersachsen', 'Bremen',
# 'Nordrhein-Westfalen', 'Hamburg', 'Hessen', 'Rheinland-Pfalz',
# 'Baden-Württemberg', 'Bayern', 'Saarland', 'Berlin', 'Brandenburg',
# 'Mecklenburg-Vorpommern', 'Sachsen', 'Thüringen', 'Sachsen-Anhalt'

age_groups = df_germany['Altersgruppe'].unique()
# 'A15-A34' 'A35-A59' 'A60-A79' 'A80+' 'A00-A04' 'A05-A14' 'unbekannt'

gender = df_germany['Geschlecht'].unique()
# 'M' 'W' 'unbekannt' --> unknown

# bayern_raw = df_germany.loc[df_germany['Bundesland'] == 'Bayern']  # 9713
# niedersachsen = df_germany.loc[df_germany['Bundesland'] == 'Niedersachsen']  # 2967
# nordrhein_westfalen = df_germany.loc[df_germany['Bundesland'] == 'Nordrhein-Westfalen']  # 6427


def sort_cumsum_data_by_state(df, state):
    df_state = df.loc[df['Bundesland'] == state]
    df_state.rename(columns={'Bundesland': 'State',
                             'AnzahlFall': 'NewCases',
                             'AnzahlTodesfall': 'DeathCases'}, inplace=True)

    state_sort_data = df_state.groupby(['State', 'RegistrationDate']).sum().reset_index()
    state_cases = pd.DataFrame(state_sort_data,
                               columns=['State', 'RegistrationDate', 'NewCases', 'DeathCases'])
    state_cases['NewCasesTotal'] = state_cases['NewCases'].cumsum()
    state_cases['DeathCasesTotal'] = state_cases['DeathCases'].cumsum()
    state = pd.DataFrame(state_cases,
                         columns=['State', 'RegistrationDate',
                                  'NewCases', 'DeathCases', 'NewCasesTotal', 'DeathCasesTotal'])
    return state


def convert_timestamp_to_string(timestamp):
    return timestamp.astype(str).str[:10]


bayern = sort_cumsum_data_by_state(df_germany, 'Bayern')
niedersachsen = sort_cumsum_data_by_state(df_germany, 'Niedersachsen')
nordrhein_westfalen = sort_cumsum_data_by_state(df_germany, 'Nordrhein-Westfalen')

date_minBY = np.datetime64(bayern['RegistrationDate'].min(), 'D')
date_maxBY = np.datetime64(bayern['RegistrationDate'].max(), 'D') + np.timedelta64(1, 'D')

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('legend', fontsize=15)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('axes', labelsize=15)

fig = plt.figure(figsize=[20, 20])
ax = fig.add_subplot(2, 1, 1)
# plt.xticks(rotation=80)
# plt.xlim(date_minBY, date_minBY)

ax.set_xlabel('Date')
ax.set_ylabel('Accumulated cases confirmed in Bayern')
ax.set_title('Accumulated cases confirmed in Bayern')
print(bayern['RegistrationDate'].iloc[-3])

# the data of the RKI is not complete within the last four days, cases will be added in the next days
# therefore I draw a line at which the interpretation ist not possible because of too less repprted cases
# ax.axvline(x=bayern['RegistrationDate'].iloc[-4], color='red', linewidth=8)
# ax.set_yscale('log')


# NRWcumSumCasesDF=dfNRW['CumSum_cases']
# ax.scatter(dfNRW.date,dfNRW.CumSum_cases,color='blue')
ax.plot(convert_timestamp_to_string(bayern['RegistrationDate']), bayern['NewCasesTotal'],
        label='BY', color='green')

plt.show()
#ax.plot(dfBayern.date, dfBayern.CumSum_death, label='Bayern')