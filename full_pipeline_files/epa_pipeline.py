'''
Group 13 -- Predicting RCRA Significant Violations
'''
import numpy as np
import pandas as pd
import combined_features_rachel as cf
import model_analyzer as ma
import ml_pipe as ml
import cleaners as cl
import sys

'''
read in and begin cleaning constituent datasets 
'''

def temp_holdout_prep(facilities_csv, evaluations_csv, violations_csv,
                     snc_csv, start_date):
    FACILITIES = pd.read_csv(facilities_csv)
    EVALS = pd.read_csv(evaluations_csv)
    cl.clean_converttodatetime_dashes(EVALS, 'EVALUATION_START_DATE', start_date)
    VIOLATIONS = pd.read_csv(violations_csv)
    cl.clean_and_converttodatetime_slashes(VIOLATIONS,
                                           'DATE_VIOLATION_DETERMINED', start_date)
    SNC = pd.read_csv(snc_csv)
    cl.yr_month_to_datetime(SNC, 'YRMONTH', start_date)

    return FACILITIES, EVALS, VIOLATIONS, SNC

#create temporal test and train features and true Y values
def temporal_split(evals_df, vios_df, snc_df):
    periodized = [ml.temp_holdout(evals_df, 'EVALUATION_START_DATE' , 24, 24),
                  ml.temp_holdout(vios_df, 'DATE_VIOLATION_DETERMINED', 24, 24),
                  ml.temp_holdout(snc_df, 'YRMONTH', 24, 24)]
    for period in periodized:
        for x in period:
            for y in x:
                print(y.head())

    return periodized

def generate_features(p, facs_df):
    train_test_with_features = []

    for period, dfs in enumerate(p[0]):
        print(p[0][period][0].head())
        train_test_with_features.append(
            cf.create_all_features(facs_df, p[0][period][0],
                                    p[1][period][0],
                                    p[2][period][0]))

    return train_test_with_features

def main():
    if len(sys.argv) != 6:
        print("Usage: analyze_projects.py \
        <facilities_filename> <evals_filename> <violations_filename> <snc_filename>\
              <start_date>, file=sys.stderr")
        sys.exit(1)

    # read in data
    start_date = pd.to_datetime(sys.argv[5])
    facs, evals, vios, snc = temp_holdout_prep(sys.argv[1], sys.argv[2],
                                               sys.argv[3], sys.argv[4],
                                               start_date)
    periods = temporal_split(evals, vios, snc)
    print('temporal_split works')
    periods = generate_features(periods, facs)
    #ok still gotta get generate_features to work herer
    print(len(periods))
    for i, x in enumerate(periods):
        if i % 2 == 0:
            name = 'train period{}'.format(i)
            x.to_csv(name + '.csv')
            print('saved: ', name)
        else:
            name = 'test period{}'.format(i)
            x.to_csv(name + '.csv')
            print('saved: ', name)





if __name__ == "__main__":
    main()


                                            


