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
                     snc_csv, sd):
    FACILITIES = pd.read_csv(facilities_csv)
    EVALS = pd.read_csv(evaluations_csv)
    EVALS = cl.clean_converttodatetime_dashes(EVALS, 'EVALUATION_START_DATE',
                                              sd)
    print('evals begin: ', EVALS['EVALUATION_START_DATE'].min())
    print('evals end: ', EVALS['EVALUATION_START_DATE'].max())
    VIOLATIONS = pd.read_csv(violations_csv)
    VIOLATIONS = cl.clean_and_converttodatetime_slashes(VIOLATIONS,
                                           'DATE_VIOLATION_DETERMINED', sd)
    print('vios begin: ', VIOLATIONS['DATE_VIOLATION_DETERMINED'].min())
    print('vios end: ', VIOLATIONS['DATE_VIOLATION_DETERMINED'].max())
    SNC = pd.read_csv(snc_csv)
    SNC = cl.yr_month_to_datetime(SNC, 'YRMONTH', sd)
    print('SNC begin: ', SNC['YRMONTH'].min())
    print('SNC end: ', SNC['YRMONTH'].max())

    return FACILITIES, EVALS, VIOLATIONS, SNC

#create temporal test and train features and true Y values
def temporal_split(evals_df, vios_df, snc_df):
    eval_train, eval_test = ml.temp_holdout(evals_df, 'EVALUATION_START_DATE' , 24, 24)
    vio_train, vio_test = ml.temp_holdout(vios_df, 'DATE_VIOLATION_DETERMINED', 24, 24)
    snc_train, snc_test = ml.temp_holdout(snc_df, 'YRMONTH', 24, 24)

    trains = [eval_train, vio_train, snc_train]
    tests = [eval_test, vio_test, snc_test]

    return trains, tests

def generate_features(p, facs_df):
    train_test_with_features = []

    for period, dfs in enumerate(p[0]):
        train_test_with_features.append(
            cf.create_all_features(facs_df, dfs,
                                    p[1][period],
                                    p[2][period]))
        train_test_with_features.append(
            cf.create_all_features(facs_df, p[3][period],
                                    p[4][period],
                                    p[5][period]))

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
    trains, tests = temporal_split(evals, vios, snc)
    print('temporal_split works')
    periods = generate_features(trains + tests, facs)
    #ok still gotta get generate_features to work herer
    print(len(periods))
    train_count = 0
    test_count = 0
    for i, x in enumerate(periods):
        if i % 2 == 0:
            train_count += 1
            name = 'train_period{}'.format(train_count)
            x.to_csv(name + '.csv')
            print('saved: ', name)
        else:
            test_count += 1
            name = 'test_period{}'.format(test_count)
            x.to_csv(name + '.csv')
            print('saved: ', name)





if __name__ == "__main__":
    main()


                                            


