"""All features from Carla, Esther, and Rachel combined"""

import sys
from datetime import datetime
from pipeline_library import *


def create_final(more_info_df, facilities_df):
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    # al = 'ACTIVITY_LOCATION'
    more_info_df = pd.merge(more_info_df, facilities_df[[ids, zips, states]], on=ids,\
        how='left')
    gb = more_info_df.groupby([ids, zips, states])\
        .size().reset_index()
    data = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    facilities_with_features_df = pd.DataFrame(data=data)
    return facilities_with_features_df, more_info_df


def time_late(violations_df, max_date, facilities_df):

    print("PASSING INTO TIME LATE")
    print(facilities_df.info())
    '''
    Inputs:
        violations_df: a dataframe with all violations information
        max_date: the maximum date in our training/testing set
    !!!DONE!!!

    Calculates:
        Number of times early/late
        Average time early/late
        Total time early/late
        Number of days since early/late

        for a sinlg location
        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1

    Filter by zip codes

    Filter by all facility ids
    '''
    #You can change these if need be
    #I am assuming all of these will be in the df passed here
    ids = 'ID_NUMBER'
    actual = 'ACTUAL_RTC_DATE'
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    diff = 'difference'
    early = 'early '
    late = 'late '
    #al = 'ACTIVITY_LOCATION'

    facilities_with_features_df, violations_df = create_final(violations_df, facilities_df)

    for col in [actual, scheduled]:
        violations_df[col] = pd.to_datetime(violations_df[col])

    violations_df[diff] = violations_df[actual] - violations_df[scheduled]
    violations_df[diff] = violations_df[diff]\
        .apply(lambda x: x.days)
    violations_df[early] = violations_df[diff]\
        .apply(lambda x: 0 if x >= 0 else 1)
    violations_df[late] = violations_df[diff]\
        .apply(lambda x: 0 if x <= 0 else 1)

    for col in [early, late]:
        filt = (violations_df[col] == 1)
        our_db = violations_df[filt]
        #for group in [ids, al]:
        for group in [ids, zips, states]:
            label = col + " " + group
            avg = our_db.groupby(group)\
                [diff].mean().reset_index().rename(\
                columns={diff:label+" avg"})
            sums = our_db.groupby(group)\
                [diff].sum().reset_index().rename(\
                columns={diff:label+" sum"})
            count = our_db.groupby(group)\
                [col].sum().reset_index().rename(\
                columns={col:label+" count"})
            last = our_db.groupby(group)\
                [actual].max()\
                .apply(lambda x: (max_date - x).days)\
                .reset_index().rename(\
                columns={actual:"last " + label})
            for gb_bool in [(avg, False), (sums, False), (count, False),\
                (last, True)]:
                gb, bool_val = gb_bool
                facilities_with_features_df = pd.merge(facilities_with_features_df, gb,\
                    on=group, how='left')
                if bool_val:
                    to_fill = "last " + label
                    facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill]\
                        .fillna(value=float('Inf'))

    return facilities_with_features_df

# OLD VERSION OF TIME_LATE
'''
def time_late(date1, date2, violations_df):
    """
    !!!DONE!!!

    Calculates:
        Number of times early/late
        Average time early/late
        Total time early/late
        Number of days since early/late

        for a sinlg location
        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1

    Filter by zip codes

    Filter by all facility ids
    """
    # You can change these if need be
    # I am assuming all of these will be in the df passed here
    ids = 'ID_NUMBER'
    date_to_split = 'ACTUAL_RTC_DATE'
    actual = date_to_split
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    diff = 'difference'
    early = 'early '
    late = 'late '
    between = ' between: ' + str(date2) + " - " + str(date1)
    before = ' before: ' + str(date1)

    violations_df[diff] = violations_df[actual] - violations_df[scheduled]
    violations_df[diff] = violations_df[diff] \
        .apply(lambda x: x.days)
    violations_df[early] = violations_df[diff] \
        .apply(lambda x: 0 if x >= 0 else 1)
    violations_df[late] = violations_df[diff] \
        .apply(lambda x: 0 if x <= 0 else 1)

    filt_between = \
        (violations_df[date_to_split] <= date1) & \
        (violations_df[date_to_split] >= date2)
    filt_before = (violations_df[date_to_split] <= date1)

    df_between = violations_df[filt_between]
    df_before = violations_df[filt_before]

    for col in [early, late]:
        for db_label in [(df_between, between), (df_before, before)]:
            db, label = db_label
            label = col + " " + label
            filt = (db[col] == 1)
            our_db = db[filt]
            for group in [ids, zips, states]:
                label += " " + group
                avg = our_db.groupby(group) \
                    [diff].mean().reset_index().rename(columns={diff: label + " avg"})
                sums = our_db.groupby(group) \
                    [diff].sum().reset_index().rename(columns={diff: label + " sum"})
                count = our_db.groupby(group) \
                    [col].sum().reset_index().rename(columns={col: label + " count"})
                last = our_db.groupby(group) \
                    [actual].max() \
                    .apply(lambda x: (date1 - x).days) \
                    .reset_index().rename(columns={actual: "last " + label})
                for gb in [avg, sums, count, last]:
                    violations_df_time_features = pd.merge(violations_df, gb, on=group, how='left')

    return violations_df_time_features
    '''


def num_inspections(evals_df, max_date, facilities_df):
    '''
    Inputs:
        facilities_df: a dataframe with all evaluations information
        max_date: the maximum date in our training/testing set

    !!!DONE!!!
    Calculates:
        Number of inspections
        Days since last inspection

        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1

    Generates features based on the number of inspections
    date1: we want evaluations before this
    date2: if we want to filter dates between two dates
    '''
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    al = 'ACTIVITY_LOCATION'

    '''
    facilities_crop = facilities_df[[ids, states, zips]].groupby([ids, states, zips]).size().reset_index()
    print(facilities_crop.head())
    print(evals_df.head())

    evals_df = pd.merge(evals_df, facilities_crop, on=ids, how='left').reset_index()
    print(evals_df.head())

    # if there haven't been any inspections, will set to the max # of possible days in dataset
    
    max_val_for_na = (evals_df[date].max() - evals_df[date].min()).days
    print("DEBUGGING NUM INSPECTS")
    # print(max_val_for_na)
    '''

    # find the number of evals by state
    #sums_state = evals_df.groupby(states).size().reset_index().rename(columns={0: "NumEvalsInMyState"}).fillna(value=0)
    #print(sums_state.head())

    #facilities_result = pd.merge(facilities_df, sums_state, on=states, how='left')

    # sums_zip = evals_df.groupby(zips).size().reset_index().rename(columns={0: "NumEvalsInMyZIP"}).fillna(value=0)
    # facilities_result = pd.merge(facilities_result, sums_zip, on=zips, how='left')

    # latest_eval_state = evals_df.loc[evals_df.groupby(states)[date].idxmin()].reset_index().rename(columns={0: "NumEvalsInMyState"}).fillna(value=max_val_for_na)
    # facilities_result = pd.merge(facilities_result, latest_eval_state, on=states, how='left')


    #print(facilities_result.info())
    #facilities_result.head().to_csv("debugging_numins.csv")

    facilities_with_features_df, evals_df = create_final(evals_df, facilities_df)
    evals_df[date] = pd.to_datetime(evals_df[date])
    
    # for group in [ids, al]:
    for group in [ids, zips, states]:
        sums = evals_df.groupby(group) \
            .size().reset_index()
        last = evals_df.groupby(group) \
            [date].max() \
            .apply(lambda x: (max_date - pd.to_datetime(x)).days) \
            .reset_index().rename(columns={date: group + " last"})
        for gb_bool in [(sums, False), (last, True)]:
            gb, bool_val = gb_bool
            facilities_with_features_df = \
                pd.merge(facilities_with_features_df, gb, on=group, how='left')\
                .rename(columns={0: "eval_sum_" + group})
            if bool_val:
                to_fill = group + " last"
                facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill] \
                    .fillna(value=float('Inf'))
    

    return facilities_with_features_df
    #return

# TODO: check in on this function
def corrective_event(date1, date2, df_all_data):
    """
    Generates features based on a corrective action event
    """
    date = 'Actual Date of Event'
    event = 'Corrective Action Event Code'

    filt_between = \
        (df_all_data[date] <= date1) & \
        (df_all_data[date] >= date2)
    filt_before = (df_all_data[date] <= date1)

    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    val_unique = df_all_data[event].unique()
    for val in val_unique:
        for df in [df_between, df_before]:
            new_col = str(val)
            df_all_data[new_col] = df_all_data[events] \
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = df_all_data.groupby(group)[new_col].sum() \
                    .reset_index() \
                    .rename(columns={new_col: new_col + group})
                df_all_data = pd.merge(df_all_data, to_merge, on=group, how='left')

    csv_name = 'Corrective_Action_Event.csv'
    df_ca = pd.read_csv(csv_name, usecols=[0, 5])
    return df_ca


# TODO: confirm which dataframe this is using?
def type_waste(df_all_data):
    """
    !!!DONE!!!

    calculates:
        dummy variable for waste code/ code owner/ naics code
            for a single facility
        calculates all facilities with waste code/ code owner/ naics code
            in a zip/state

    Generates features based on the type of waste created
    """
    # csv_name = 'Biennial_Report_GM_Waste_Code.csv'
    # df_wc = pd.read_csv(csv_name, header=[0,6])
    # I'm assuming the the three below columns will all be merged into the db
    waste_codes = 'Hazardous Waste Code'
    code_owner = 'Hazardous Waste Code Owner'
    naics = 'NAICS_CODE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    for col in [waste_codes, code_owner, naics]:
        ser = df_all_data[col]
        val_unique = ser.unique()
        for val in val_unique:
            new_col = col + str(val)
            df_all_data[new_col] = df_all_data[waste_codes] \
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = df_all_data.groupby(group)[new_col].sum() \
                    .reset_index() \
                    .rename(columns={new_col: new_col + group})
                df_all_data = pd.merge(df_all_data, to_merge, on=group, how='left')

    return df_all_data


def num_facilities(facilities_df):
    """
    !!!DONE!!!

    Calculates:
        Number of facilities in a zip code and state
    """
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    sums_state = facilities_df.groupby(states).size().reset_index().rename(columns={0: "NumInMyState"})
    facilities_w_num_nearby = pd.merge(facilities_df, sums_state, on=states, how='left')

    # fill na values with zero
    facilities_w_num_nearby["NumInMyState"].fillna(0, inplace=True)

    sums_zip = facilities_w_num_nearby.groupby(zips).size().reset_index().rename(columns={0: "NumInMyZIP"})
    facilities_w_num_nearby = pd.merge(facilities_w_num_nearby, sums_zip, on=zips, how='left')
    facilities_w_num_nearby["NumInMyZIP"].fillna(0, inplace=True)

    return facilities_w_num_nearby


def create_eval_features(facilities_df, evals_df):
    """Takes in data frames and returns a single data frame for running models."""
    evals_counts = evals_df.groupby("ID_NUMBER").size().reset_index().rename(columns={
        0: 'EvalCount'}).sort_values("EvalCount", ascending=False)

    # facilities_w_counts = pd.merge(evals_counts, facilities_df, left_on='ID_NUMBER', right_on='ID_NUMBER', how='left')
    facilities_w_counts = pd.merge(facilities_df, evals_counts, left_on='ID_NUMBER', right_on='ID_NUMBER', how='outer')

    # fill na values of EvalCounts with zero, since that means no evaluations in the data
    facilities_w_counts["EvalCount"].fillna(0, inplace=True)

    evals_df["FOUND_VIOLATION_01"] = pd.Series(np.where(evals_df.FOUND_VIOLATION.str.contains("Y"), 1, 0),
                                               evals_df.index)

    violation_sums = evals_df.groupby("ID_NUMBER")['FOUND_VIOLATION_01'].sum().reset_index().rename(
        columns={'FOUND_VIOLATION_01': 'Sum_Violations'}).sort_values("Sum_Violations", ascending=False)

    facilities_w_violations = pd.merge(facilities_w_counts, violation_sums, left_on='ID_NUMBER', right_on='ID_NUMBER',
                                       how='left')

    # fill na values of Sum_Violations with zero, since that means no violations in the data
    facilities_w_violations['Sum_Violations'].fillna(0, inplace=True)

    facilities_w_violations["PCT_EVALS_FOUND_VIOLATION"] = facilities_w_violations["Sum_Violations"] / \
                                                           facilities_w_violations["EvalCount"]

    # fill na values of PCT_EVALS_FOUND_VIOLATION with zero, since that means no evals
    facilities_w_violations["PCT_EVALS_FOUND_VIOLATION"].fillna(0, inplace=True)

    facilities_w_violations["PCT_OF_ALL_EVALS"] = facilities_w_violations["EvalCount"] / facilities_w_violations[
        "EvalCount"].sum()

    facilities_w_violations["PCT_OF_ALL_VIOLATIONS"] = facilities_w_violations["Sum_Violations"] / \
                                                       facilities_w_violations["Sum_Violations"].sum()

    max_date = evals_df.groupby("ID_NUMBER").agg({'EVALUATION_START_DATE': 'max'})[
        ['EVALUATION_START_DATE']].reset_index().rename(columns={'EVALUATION_START_DATE': 'MostRecentEval'})

    facilities_w_violations = pd.merge(facilities_w_violations, max_date, left_on='ID_NUMBER', right_on='ID_NUMBER',
                                       how='left')

    facilities_w_violations["MostRecentEval"] = pd.to_datetime(facilities_w_violations["MostRecentEval"],
                                                               format='%m/%d/%Y', errors='coerce')

    facilities_w_violations["NumMonthsSinceEval"] = (datetime.today() - facilities_w_violations["MostRecentEval"]) / (
        np.timedelta64(1, 'M'))

    evals_df["date_version"] = pd.to_datetime(evals_df["EVALUATION_START_DATE"], format='%m/%d/%Y',
                                                        errors='coerce')

    max_poss_months = ((datetime.today() - (evals_df["date_version"].min())) / (np.timedelta64(1, 'M')))

    # fill the na values with the maximum possible number of months since evaluation
    facilities_w_violations["NumMonthsSinceEval"].fillna(max_poss_months, inplace=True)

    # drop the date column before you return
    facilities_w_violations = facilities_w_violations.drop(['MostRecentEval'], axis=1)

    return facilities_w_violations


def flag_lqg(facilities_df):
    """Adds a column to the data frame that is 1 if the facility is an LQG."""
    # 3 is the code for LQG according to the data documentation at
    #   https://echo.epa.gov/tools/data-downloads/rcrainfo-download-summary#download_files
    facilities_df["IsLQG"] = pd.Series(np.where(facilities_df.FED_WASTE_GENERATOR.str.contains("3"), 1, 0), \
                                       facilities_df.index)
    facilities_df["IsTSDF"] = pd.Series(np.where(facilities_df.HREPORT_UNIVERSE_RECORD.str.contains("TSDF"), 1, 0), \
                                       facilities_df.index)

    return facilities_df


def snc_info(facilities_df, snc_df):
    """Adds columns for info about SNC status."""
    # TODO: Need to add ANY SNC in time period
    snc_df["SNC_Y"] = np.where(snc_df.SNC_FLAG.str.contains("Y"), 1, 0)
    snc_df["SNC_N"] = pd.Series(np.where(snc_df.SNC_FLAG.str.contains("N"), 1, 0), snc_df.index)

    snc_y = snc_df[snc_df['SNC_Y'] == 1]
    snc_n = snc_df[snc_df['SNC_N'] == 1]

    snc_count = snc_y.groupby("ID_NUMBER").size().reset_index().rename(columns={
        0: 'SNC_Count'}).sort_values("SNC_Count", ascending=False)

    facilities_df = pd.merge(facilities_df, snc_count, on="ID_NUMBER", how="left")
    facilities_df["SNC_Count"].fillna(0, inplace=True)

    # Find the most recent dates the facility was designated an SNC, Y or N
    max_date_y = snc_y.groupby("ID_NUMBER").agg({'YRMONTH': max})[
        ['YRMONTH']].reset_index().rename(columns={'YRMONTH': 'MostRecentSNC_Y'})

    max_date_n = snc_n.groupby("ID_NUMBER").agg({'YRMONTH': 'max'})[
        ['YRMONTH']].reset_index().rename(columns={'YRMONTH': 'MostRecentSNC_N'})

    facilities_df = pd.merge(facilities_df, max_date_y, on='ID_NUMBER', how='left')
    facilities_df = pd.merge(facilities_df, max_date_n, on='ID_NUMBER', how='left')

    facilities_df["MostRecentSNC_Y"] = pd.to_datetime(facilities_df["MostRecentSNC_Y"],
                                                               format='%Y%m', errors='coerce')
    facilities_df["MostRecentSNC_N"] = pd.to_datetime(facilities_df["MostRecentSNC_N"],
                                                   format='%Y%m', errors='coerce')

    facilities_df["More_Recent_SNC_Yes"] = np.where((facilities_df['MostRecentSNC_Y'] >= facilities_df['MostRecentSNC_N']), 1, 0)

    # Drop the date columns before returning df
    facilities_df = facilities_df.drop(['MostRecentSNC_Y', 'MostRecentSNC_N'], axis=1)
    print("FINAL INFO FROM snc_info")
    print(facilities_df.info())

    return facilities_df


def create_outcome_vars(facilities_df):
    """Adds dummy columns to the data frame for if there was a violation and if there was a serious violation."""
    facilities_df["HasViolation"] = pd.Series(np.where(np.equal(facilities_df.Sum_Violations, 0), 0, 1), \
                                       facilities_df.index)

    facilities_df["HadAnySNC"] = pd.Series(np.where(np.equal(facilities_df.SNC_Count, 0), 0, 1), \
                                              facilities_df.index)

    facilities_df["Currently_SNC"] = pd.Series(np.where(np.equal(facilities_df.More_Recent_SNC_Yes, 1), 1, 0), \
                                       facilities_df.index)
    return facilities_df


def create_all_features(facilities_df, evals_df, violations_df, snc_df):
    """Takes in dataframes, already trimmed into a training or test set, and returns with features added."""
    """
    Things that are definitely working:
        create_eval_features
        flag_lqg
        snc_info
        create_outcome_vars
        num_facilities
        
    
    """

    facilities_w_violations = create_eval_features(facilities_df, evals_df)
    facilities_lqg = flag_lqg(facilities_w_violations)
    facilities_snc = snc_info(facilities_lqg, snc_df)
    facilities_outcomes = create_outcome_vars(facilities_snc)
    # print(facilities_outcomes.info())
    facilities_nearby_nums = num_facilities(facilities_outcomes)

    max_date = datetime(2000, 1, 1, 0, 0)


    # TODO: THESE ARE NOT WORKING
    #(From Esther:) I got this to work on a small set of data
    facilities_w_time_late = time_late(violations_df, max_date, facilities_nearby_nums)
    facilities_w_num_ins_nearby = num_inspections(evals_df, max_date, facilities_nearby_nums)
    facilities_nearby_nums = pd.merge(facilities_nearby_nums, facilities_w_num_ins_nearby, on="ID_NUMBER", how="left")
    facilities_nearby_nums = pd.merge(facilities_nearby_nums, facilities_w_time_late, on="ID_NUMBER", how="left")

    # ADDING ACS FEATURES
    # read_data('evals')

    # print(facilities_w_time_late.info())
    # return facilities_w_time_late

    # print(facilities_nearby_nums.info())
    # return facilities_nearby_nums

    print(facilities_nearby_nums.info())
    return facilities_nearby_nums


def main():
    if len(sys.argv) != 5:
        print("Usage: analyze_projects.py \
        <facilities_filename> <evals_filename> <violations_filename> <snc_filename>", file=sys.stderr)
        sys.exit(1)

    start_date = "01-01-2012"
    end_date = "01-01-2019"

    # read in data
    facilities_df = read_data(sys.argv[1])
    evals_df = read_data(sys.argv[2])
    violations_df = read_data(sys.argv[3])
    snc_df = read_data(sys.argv[4])

    full_features_df = create_all_features(facilities_df, evals_df, violations_df, snc_df)
    full_features_df.to_csv('full_features.csv')

    print(full_features_df.info())
    print(full_features_df.sort_values("SNC_Count", ascending=False).head(20))


if __name__ == "__main__":
    main()
