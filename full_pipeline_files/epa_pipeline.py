'''
Group 13 -- Predicting RCRA Significant Violations
'''
import numpy as np
import pandas as pd
import combined_features as cf
import model_analyzer as ma
import ml_loop as ml
import clean_evals as ce

'''
read in constituent datasets 
'''

FACILITIES = pd.read_csv('RCRA_FACILITIES.csv')
EVALS = pd.read_csv('RCRA_EVALUATIONS.csv')
EVALS[date_cols[0]] = pd.to_datetime(EVALS.EVALUATION_START_DATE,
                                    format='%m/%d/%Y', errors='coerce')
ce.clean_evals(EVALS)
VIOLATIONS = pd.read_csv('RCRA_VIOLATIONS.csv')
ce.clean_vios(VIOLATIONS)
SNC = pd.read_csv('RCRA_VIOSNC_HISTORY.csv')
ce.clean_snc(SNC)
date_cols = ['EVALUATION_START_DATE', 'DATE_VIOLATION_DETERMINED', 'YRMONTH']

raw_data = [EVALS, VIOLATIONS, SNC]
periodized = [ml.temp_holdout(EVALS, date_cols[0], 24, 24),
              ml.temp_holdout(VIOLATIONS, date_cols[1], 24, 24),
              ml.temp_holdout(SNC, date_cols[2], 24, 24)]

features = []

for period, dfs in enumerate(periodized[0]):
    if dfs.EVALUATION_START_DATE.min() >= pd.to_datetime('2012-01-01'):
        features.append(cf.create_all_features(FACILITIES, periodized[0][period],
                                              periodized[1][period],
                                               periodized[2][period]))

