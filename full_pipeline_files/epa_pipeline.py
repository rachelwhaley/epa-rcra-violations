'''
Group 13 -- Predicting RCRA Significant Violations
'''
import numpy as np
import pandas as pd
import combined_features as cf
import model_analyzer as ma
import ml_loop as ml

'''
read in constituent datasets 
'''

FACILITIES = pd.read_csv('RCRA_FACILITIES.csv')
EVALS = pd.read_csv('RCRA_EVALUATIONS.csv')
EVALS[date_cols[0]] = pd.to_datetime(EVALS.EVALUATION_START_DATE,
                                    format='%m/%d/%Y', errors='coerce')
wrong_date_1 = pd.to_datetime('1920-01-01')
EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=100)
                                        if x < wrong_date_1 else x)
wrong_date_2 = pd.to_datetime('1930-01-01')
EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=60)
                                        if x < wrong_date_2 else x)
wrong_date_3 = pd.to_datetime('1940-01-01')
EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=50)
                                        if x < wrong_date_3 else x)
wrong_date_4 = pd.to_datetime('1960-01-01')
EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=30)
                                        if x < wrong_date_4 else x)
wrong_date_5 = pd.to_datetime('1970-01-01')
EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=20)
                                        if x < wrong_date_5 else x)
wrong_date_6 = pd.to_datetime('1980-01-01')
EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=10)
                                        if x < wrong_date_6 else x)


VIOLATIONS = pd.read_csv('RCRA_VIOLATIONS.csv')

SNC = pd.read_csv('RCRA_VIOSNC_HISTORY.csv', date_cols[2])
SNC['YRMONTH'] = pd.to_datetime(SNC['YRMONTH'], format='%Y%m', errors='coerce')
date_cols = ['EVALUATION_START_DATE', 'DATE_VIOLATION_DETERMINED', 'YRMONTH']

raw_data = [EVALS, VIOLATIONS, SNC]
periodized = [ml.temp_holdout(EVALS, date_cols[0], 24, 24),
              ml.temp_holdout(VIOLATIONS, date_cols[1], 24, 24),
              ml.temp_holdout(SNC, date_cols[2], 24, 24)]

features = []

for period in periodized:
    features.append(cf.create_all_features(FACILITIES, period[0],
                                           period[1], period[2]))

