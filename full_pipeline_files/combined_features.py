"""All features from Carla, Esther, and Rachel combined"""

import sys
from datetime import datetime
from pipeline_library import *


def create_final(facilities_df):
    """Facility frame with just ID, State, ZIP"""
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    #al = 'ACTIVITY_LOCATION'
    gb = facilities_df.groupby([ids, zips, states])\
        .size().reset_index()
    d = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    #gb = facilities_df.groupby([ids, al])\
        #.size().reset_index()
    #d = {ids: gb[ids], al: gb[al]}
    facilities_with_features_df = pd.DataFrame(data=d)
    return facilities_with_features_df


def time_late(violations_df, facilities_df, max_date):
    '''
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
    # You can change these if need be
    # I am assuming all of these will be in the df passed here
    ids = 'ID_NUMBER'
    actual = 'ACTUAL_RTC_DATE'
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    diff = 'difference'
    early = 'early '
    late = 'late '
    # al = 'ACTIVITY_LOCATION'

    facilities_with_features_df = create_final(facilities_df)

    violations_df[actual] = pd.to_datetime(violations_df[actual],
                                           format='%m/%d/%Y', errors='coerce')
    violations_df[scheduled] = pd.to_datetime(violations_df[scheduled],
                                           format='%m/%d/%Y', errors='coerce')

    facilities_df[diff] = violations_df[actual] - violations_df[scheduled]
    facilities_df[actual] = violations_df[actual]
    facilities_df[scheduled] = violations_df[scheduled]
    facilities_df[diff] = facilities_df[diff] \
        .apply(lambda x: x.days)
    facilities_df[early] = facilities_df[diff] \
        .apply(lambda x: 0 if x >= 0 else 1)
    facilities_df[late] = facilities_df[diff] \
        .apply(lambda x: 0 if x <= 0 else 1)
    facilities_df[[ids, zips, states]].ffill()

    gb = facilities_df.groupby([ids, zips, states]) \
        .size().reset_index()
    d = {ids: gb[ids], zips: gb[zips], states: gb[states]}
    # gb = facilities_df.groupby([ids, al])\
    # .size().reset_index()
    # d = {ids: gb[ids], al: gb[al]}
    factlities_with_features_df = pd.DataFrame(data=d)

    for col in [early, late]:
        filt = (facilities_df[col] == 1)
        our_db = facilities_df[filt]
        # for group in [ids, al]:
        for group in [ids, zips, states]:
            label = col + " " + group
            avg = our_db.groupby(group) \
                [diff].mean().reset_index().rename( \
                columns={diff: label + " avg"})
            sums = our_db.groupby(group) \
                [diff].sum().reset_index().rename( \
                columns={diff: label + " sum"})
            count = our_db.groupby(group) \
                [col].sum().reset_index().rename( \
                columns={col: label + " count"})
            #the trouble is in how this is being called (I changed max to
            # nanmax) but max was also not working
            last = our_db.groupby(group) \
                [actual].nanmax() \
                .apply(lambda x: (max_date - x).days) \
                .reset_index().rename( \
                columns={actual: "last " + label})
            for gb_bool in [(avg, False), (sums, False), (count, False), \
                            (last, True)]:
                gb, bool_val = gb_bool
                facilities_with_features_df = pd.merge(facilities_with_features_df, gb, \
                                                       on=group, how='left')
                if bool_val:
                    to_fill = "last " + label
                    factlities_with_features_df[to_fill] = facilities_with_features_df[to_fill] \
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


def num_inspections(evals_df, facilities_df):
    """
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
    """

    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    ids = 'ID_NUMBER'

    evals_w_geo = pd.merge(evals_df, facilities_df[[ids, states, zips]], on="ID_NUMBER", how="left")
    print(evals_w_geo.head())

    sums_state = evals_w_geo.groupby(states).size()  # .reset_index() # .rename(columns={0: "NumEvalsInMyState"})
    facilities_w_num_nearby = pd.merge(facilities_df, sums_state, on=states, how='left')

    sums_zip = facilities_w_num_nearby.groupby(zips).size().reset_index().rename(columns={0: "NumEvalsInMyZIP"})
    facilities_w_num_nearby = pd.merge(facilities_w_num_nearby, sums_zip, on=zips, how='left')

    return facilities_w_num_nearby

'''
    # original code
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    
    filt_between = \
        (evals_df[date] <= date1) & \
        (evals_df[date] >= date2)
    filt_before = (evals_df[date] <= date1)

    df_between = evals_df[filt_between]
    df_before = evals_df[filt_before]

    for group in [ids, zips, states]:
    # for group in [ids, 'ACTIVITY_LOCATION']:
        for db in evals_df:
            sums = db.groupby(group) \
                .size().reset_index()
            last = db.groupby(group) \
                [date].max() \
                .apply(lambda x: (date1 - x).days) \
                .reset_index()
            for gb in [sums, last]:
                evals_grouped_num_inspections = pd.merge(evals_df, gb, on=group, how='left')

    print(evals_grouped_num_inspections.head())

    return evals_grouped_num_inspections # .groupby("ID_NUMBER").size().reset_index() # .rename(columns={0: "NumInMyState"})
    '''

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

    sums_zip = facilities_w_num_nearby.groupby(zips).size().reset_index().rename(columns={0: "NumInMyZIP"})
    facilities_w_num_nearby = pd.merge(facilities_w_num_nearby, sums_zip, on=zips, how='left')

    return facilities_w_num_nearby


def create_eval_features(facilities_df, evals_df):
    """Takes in data frames and returns a single data frame for running models."""
    evals_counts = evals_df.groupby("ID_NUMBER").size().reset_index().rename(columns={
        0: 'EvalCount'}).sort_values("EvalCount", ascending=False)

    # facilities_w_counts = pd.merge(evals_counts, facilities_df, left_on='ID_NUMBER', right_on='ID_NUMBER', how='left')
    facilities_w_counts = pd.merge(facilities_df, evals_counts, left_on='ID_NUMBER', right_on='ID_NUMBER', how='outer')

    evals_df["FOUND_VIOLATION_01"] = pd.Series(np.where(evals_df.FOUND_VIOLATION.str.contains("Y"), 1, 0),
                                               evals_df.index)

    violation_sums = evals_df.groupby("ID_NUMBER")['FOUND_VIOLATION_01'].sum().reset_index().rename(
        columns={'FOUND_VIOLATION_01': 'Sum_Violations'}).sort_values("Sum_Violations", ascending=False)

    facilities_w_violations = pd.merge(facilities_w_counts, violation_sums, left_on='ID_NUMBER', right_on='ID_NUMBER',
                                       how='left')

    facilities_w_violations["PCT_EVALS_FOUND_VIOLATION"] = facilities_w_violations["Sum_Violations"] / \
                                                           facilities_w_violations["EvalCount"]

    facilities_w_violations["PCT_OF_ALL_EVALS"] = facilities_w_violations["EvalCount"] / facilities_w_violations[
        "EvalCount"].sum()

    facilities_w_violations["PCT_OF_ALL_VIOLATIONS"] = facilities_w_violations["Sum_Violations"] / \
                                                       facilities_w_violations["Sum_Violations"].sum()

    max_date = evals_df.groupby("ID_NUMBER").agg({'EVALUATION_START_DATE': 'max'})[
        ['EVALUATION_START_DATE']].reset_index().rename(columns={'EVALUATION_START_DATE': 'MostRecentEval'})

    facilities_w_violations = pd.merge(facilities_w_violations, max_date, left_on='ID_NUMBER', right_on='ID_NUMBER',
                                       how='left')


    facilities_w_violations["NumMonthsSinceEval"] = (datetime.today() - facilities_w_violations["MostRecentEval"]) / (
        np.timedelta64(1, 'M'))

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
    snc_df["SNC_Y"] = pd.Series(np.where(snc_df.SNC_FLAG.str.contains("Y"), 1, 0), snc_df.index)

    snc_df["SNC_N"] = pd.Series(np.where(snc_df.SNC_FLAG.str.contains("N"), 1, 0), snc_df.index)
    print(snc_df.head())

    # gapminder['year']==2002
    snc_y = snc_df[snc_df['SNC_Y'] == 1]
    snc_n = snc_df[snc_df['SNC_N'] == 1]
    print(snc_y.info())

    # snc_n = snc_df.SNC_FLAG.str.contains("N")

    snc_count = snc_y.groupby("ID_NUMBER").size().reset_index().rename(columns={
        0: 'SNC_Count'}).sort_values("SNC_Count", ascending=False)

    facilities_df = pd.merge(facilities_df, snc_count, on="ID_NUMBER", how="left")

    max_date_y = snc_y.groupby("ID_NUMBER").agg({'YRMONTH': 'max'})[
        ['YRMONTH']].reset_index().rename(columns={'YRMONTH': 'MostRecentSNC_Y'})

    max_date_n = snc_n.groupby("ID_NUMBER").agg({'YRMONTH': 'max'})[
        ['YRMONTH']].reset_index().rename(columns={'YRMONTH': 'MostRecentSNC_N'})

    facilities_df = pd.merge(facilities_df, max_date_y, on='ID_NUMBER', how='left')
    facilities_df = pd.merge(facilities_df, max_date_n, on='ID_NUMBER', how='left')

    facilities_df["MostRecentSNC_Y"] = pd.to_datetime(facilities_df["MostRecentSNC_Y"],
                                                               format='%Y%m', errors='coerce')
    facilities_df["MostRecentSNC_N"] = pd.to_datetime(facilities_df["MostRecentSNC_N"],
                                                   format='%Y%m', errors='coerce')

    facilities_df["More_Recent_SNC_Yes"] = facilities_df["MostRecentSNC_Y"] > facilities_df["MostRecentSNC_N"]

    print(facilities_df.info())

    return facilities_df


def create_outcome_vars(facilities_df):
    """Adds dummy columns to the data frame for if there was a violation and if there was a serious violation."""
    facilities_df["HasViolation"] = pd.Series(np.where(np.equal(facilities_df.Sum_Violations, 0), 0, 1), \
                                       facilities_df.index)

    facilities_df["HadAnySNC"] = pd.Series(np.where(np.equal(facilities_df.SNC_Count, 0), 0, 1), \
                                              facilities_df.index)

    facilities_df["Currently_SNC"] = pd.Series(np.where(np.equal(facilities_df.More_Recent_SNC_Yes, True), 1, 0), \
                                       facilities_df.index)
    return facilities_df


def create_all_features(facilities_df, evals_df, violations_df, snc_df):
    """Takes in dataframes, already trimmed into a training or test set, and returns with features added."""

    facilities_w_violations = create_eval_features(facilities_df, evals_df)
    facilities_lqg = flag_lqg(facilities_w_violations)
    facilities_snc = snc_info(facilities_lqg, snc_df)
    facilities_outcomes = create_outcome_vars(facilities_snc)
    # print(facilities_outcomes.info())
    facilities_nearby_nums = num_facilities(facilities_outcomes)

    max_date = datetime(2000, 1, 1, 0, 0)

    facilities_w_time_late = time_late(violations_df, facilities_nearby_nums, max_date)
    # facilities_w_num_ins_nearby = num_inspections(evals_df, facilities_nearby_nums)
    # facilities_nearby_nums = pd.merge(facilities_nearby_nums, facilities_w_num_ins_nearby, on="ID_NUMBER", how="left")
    return facilities_w_time_late


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
