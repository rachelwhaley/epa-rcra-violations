'''
Group 13 -- Predicting RCRA Significant Violations
'''
import numpy as np
import pandas as pd
import combined_features as cf
import model_analyzer as ma
import ml_loop as ml
import cleaners as cl
import sys

'''
read in and begin cleaning constituent datasets 
'''

def temp_holdout_prep(facilities_csv, evaluations_csv, violations_csv,
                     snc_csv):
    FACILITIES = pd.read_csv(facilities_csv)
    EVALS = pd.read_csv(evaluations_csv)
    cl.clean_converttodatetime_dashes(EVALS, 'EVALUATION_START_DATE')
    VIOLATIONS = pd.read_csv(violations_csv)
    cl.clean_and_convert_to_datetime_slashes(VIOLATIONS, 'DATE_VIOLATION_DETERMINED')
    SNC = pd.read_csv(snc_csv)
    cl.yr_month_to_datetime(SNC, 'YRMONTH')
    print(SNC.head())

    return FACILITIES, EVALS, VIOLATIONS, SNC

#create temporal test and train features and true Y values
def temporal_split(evals_df, vios_df, snc_df):
    periodized = [ml.temp_holdout(evals_df, 'EVALUATION_START_DATE' , 24, 24),
                  ml.temp_holdout(vios_df, 'DATE_VIOLATION_DETERMINED', 24, 24),
                  ml.temp_holdout(snc_df, 'YRMONTH', 24, 24)]
    for period in periodized:
        for x in period:
            print(x.head())

    return periodized

def generate_features(p):
    train_test_with_features = []

    for period, dfs in enumerate(p[0]):
        if dfs.EVALUATION_START_DATE.min() >= pd.to_datetime('2012-01-01'):
            train_test_with_features.append(
                cf.create_all_features(FACILITIES, p[0][period],
                                       p[1][period],
                                       p[2][period]))

    return train_test_with_features

def main():
    if len(sys.argv) != 5:
        print("Usage: analyze_projects.py \
        <facilities_filename> <evals_filename> <violations_filename> <snc_filename>", file=sys.stderr)
        sys.exit(1)

    # read in data
    facs, evals, vios, snc = temp_holdout_prep(sys.argv[1], sys.argv[2],
                                               sys.argv[3], sys.argv[4])
    periods = temporal_split(evals, vios, snc)
    print('temporal_split works')
    periods = generate_features(periods)
    #ok still gotta get generate_features to work herer
    print(periods)





if __name__ == "__main__":
    main()


                                            


