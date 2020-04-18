import numpy as np  # linear algebra
import pandas as pd
import etl.ExtractCleanAge as etl
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import dates as dt

import matplot_date.CreateMatplotXdata as matplt

# Data Clean
df_patient = pd.read_csv('/Users/hou0002j/Projekt/covid-19/dataset/COVID19_open_line_list.csv')
# clean sex coulumn for misspellings
df_patient.loc[df_patient['sex'] == 'Female'] = 'female'
df_patient.loc[df_patient['sex'] == 'Male'] = 'male'

gender_filter_male = df_patient['sex'] == 'male'
gender_filter_female = df_patient['sex'] == 'female'

# COVID 19 infected age groups
china_filter = df_patient['country'] == 'China'
df_china = df_patient[china_filter]
df_china_male = df_patient.loc[china_filter & gender_filter_male, :]
df_china_female = df_patient.loc[china_filter & gender_filter_female, :]


df_china_male_clean = etl.ExtractCleanAge(df_china_male, df_china_male['age'])
df_china_male_age_avg = df_china_male_clean.cal_age_avg()

korea_filter = df_patient['country'] == 'South Korea'
df_korea = df_patient.loc[korea_filter, :]

italy_filter = df_patient['country'] == 'Italy'

# COVID 19 Sex distribution
df_patient_sex = df_patient[['sex']].groupby(["sex"]).count()



