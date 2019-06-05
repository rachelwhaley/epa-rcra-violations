'''
Group 13 -- Predicting RCRA Significant Violations
'''
import numpy as np
import pandas as pd
import combined_features_rachel as cf
import model_analyzer as ma
import ml_pipe as ml
import cleaners as cl
import grids as g
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
    eval_train, eval_test, etr_ends, ete_ends = ml.temp_holdout(evals_df,
                                                              'EVALUATION_START_DATE', 24, 24)
    vio_train, vio_test, vtr_ends, vte_ends = ml.temp_holdout(vios_df, 'DATE_VIOLATION_DETERMINED', 24, 24)
    snc_train, snc_test, str_ends, ste_ends = ml.temp_holdout(snc_df, 'YRMONTH', 24, 24)

    trains = [eval_train, vio_train, snc_train]
    tests = [eval_test, vio_test, snc_test]
    train_ends = etr_ends
    test_ends = ete_ends

    return trains, tests, train_ends, test_ends

def generate_features(trains, tests, train_ends, test_ends, facs_df):
    p = trains + tests
    trains = []
    tests = []

    for period, dfs in enumerate(p[0]):

        trains.append(
            cf.create_all_features(facs_df, dfs, p[1][period], p[2][period],
                                   train_ends[period]))
        tests.append(
            cf.create_all_features(facs_df, p[3][period], p[4][period],
                                   p[5][period], test_ends[period]))

    return trains, tests

def run_models(grid_size, plots, thresholds, list_of_trainx, list_of_trainy,
               list_of_testx, list_of_testy):
    '''
    takes features and y data for all train and test periods and fits/runs all
    models on grid on all
    '''
    clfs = g.clfs0
    grid = g.grid0

    predictions, models, metrics = ml.model_analyzer_over_time(clfs, grid,
                                                               plots, 
                                                               thresholds,
                                                               list_of_trainx,
                                                               list_of_trainy,
                                                               list_of_testx,
                                                               list_of_testy)

    master_metrics = pd.DataFrame(columns=list(metrics[0].columns))

    for df in metrics:
        master_metrics = pd.concat([master_metrics, df], axis=0)

    return predictions, models, nw.rank(master_metrics, 'model', 'precision_0.2pct')

def main():
    if len(sys.argv) != 6:
        print("Usage: analyze_projects.py \
        <facilities_filename> <evals_filename> <violations_filename> <snc_filename>\
              <acs_data_filename> <start_date>, file=sys.stderr")
        sys.exit(1)

    # read in data
    start_date = pd.to_datetime(sys.argv[5])
    facs, evals, vios, snc = temp_holdout_prep(sys.argv[1], sys.argv[2],
                                               sys.argv[3], sys.argv[4],
                                               start_date)
    trains, tests = temporal_split(evals, vios, snc)
    print('temporal_split works')
    periods = generate_features(trains + tests, facs)
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
