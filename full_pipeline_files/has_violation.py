'''
Takes in violations db and returns whether or not a facility
has had a violation within the time frame
'''
import pandas as pd
import numpy as np
import datetime
import cleaners
import numpy as np


def has_violation(facilities_df, violations_df, start_year=2011, end_year=2018):
    '''
    facilities: list of facilities we want to know whether or not they have had a violation
    start_year: the first year we want to know about
    end_year: the last year we want to know about
    '''   
    fac_id = 'ID_NUMBER'
    eval_year = 'YEAR_EVALUATED'
    next_year = 'NEXT_YEAR'
    has_vio = 'HasViolation'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    date = 'DATE_VIOLATION_DETERMINED'
    num_copies = end_year - start_year
    y = start_year
    id_lst = []
    year_lst = []
    years = []

    while y <= end_year:
        years.append(y)
        y += 1
    for fid in facilities_df[fac_id]:
        for y in years:
            id_lst.append(fid)
            year_lst.append(y)

    data = {fac_id: id_lst, eval_year: year_lst}
    facs_by_year = pd.DataFrame(data=data)


    facilities_with_features_df, violations_df = combine(violations_df, facilities_df)

    violations_df[has_vio] = 1
    facs_by_year = pd.merge(facs_by_year, violations_df[[has_vio,fac_id, eval_year]],\
        left_on=[fac_id, eval_year], right_on=[fac_id, eval_year], how='left')
    
    facs_by_year[has_vio].fillna(0, inplace=True)
    violations_df[next_year] = violations_df[eval_year] + 1

    facs_by_year = facs_by_year.drop_duplicates(subset=[fac_id, eval_year])

    prev_vios = pd.DataFrame()
    for y in years:
        vios_this_yr = violations_df[violations_df[next_year] == y]
        max_date = datetime.datetime(y, 1, 1, 0, 0)
        for group in [fac_id, zips, states]:
            label = "_VIOLATIONS_IN_" + group
            count = vios_this_yr.groupby(group)\
                [has_vio].sum().reset_index().rename(\
                columns={has_vio:"NUMBER"+label})
            last = vios_this_yr.groupby(group)\
                [date].max()\
                .apply(lambda x: (max_date - x).days)\
                .reset_index().rename(\
                columns={date:"DAYS_SINCE" + label})
            for gb in [count, last]:
                vios_this_yr = pd.merge(vios_this_yr, gb, on=group, how='left')
            prev_vios = pd.concat([prev_vios, vios_this_yr], ignore_index=True)

    facs_by_year = pd.merge(facs_by_year, prev_vios[[fac_id,'DAYS_SINCE_VIOLATIONS_IN_ID_NUMBER',\
       'DAYS_SINCE_VIOLATIONS_IN_STATE_CODE',\
       'DAYS_SINCE_VIOLATIONS_IN_ZIP_CODE',\
       'NUMBER_VIOLATIONS_IN_ID_NUMBER', 'NUMBER_VIOLATIONS_IN_STATE_CODE',\
       'NUMBER_VIOLATIONS_IN_ZIP_CODE',next_year]], left_on=[fac_id, eval_year],\
        right_on=[fac_id, next_year], how='left')

    for col in list(facs_by_year.columns):
        if col.startswith('NUMBER'):
            facs_by_year[col] = facs_by_year[col].fillna(value=0)
        elif col.startswith('DAYS'):
            facs_by_year[col] = facs_by_year[col].fillna(value=-1)

    return facs_by_year.drop(columns=next_year), years


def flag_lqg(facilities_df):
    """Adds a column to the data frame that is 1 if the facility is an LQG."""
    # 3 is the code for LQG according to the data documentation at
    #   https://echo.epa.gov/tools/data-downloads/rcrainfo-download-summary#download_files
    facilities_df["IsLQG"] = pd.Series(np.where(facilities_df.FED_WASTE_GENERATOR.str.contains("3"), 1, 0), \
                                       facilities_df.index)
    facilities_df["IsTSDF"] = pd.Series(np.where(facilities_df.HREPORT_UNIVERSE_RECORD.str.contains("TSDF"), 1, 0), \
                                        facilities_df.index)

    return facilities_df

def num_facilities(facilities_df):
    """
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

def combine(more_info_df, facilities_df):
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
    return facilities_with_features_df.drop_duplicates(subset=[ids]), more_info_df

def time_late_early(violations_df, max_date, facilities_df):
    '''
    Calculates:
        Number of times early/late
        Average time early/late
        Total time early/late
        Number of days since early/late

        for a sinlg location
        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1

    '''
    ids = 'ID_NUMBER'
    actual = 'ACTUAL_RTC_DATE'
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    diff = 'difference'
    early = 'early '
    late = 'late '

    facilities_with_features_df, violations_df = combine(violations_df, facilities_df) 

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
                        .fillna(value=float(-1))
                    facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill]\
                        .apply(lambda x: x if x >= 0 else 0)

    return facilities_with_features_df.drop(columns=[zips,states])

def num_inspections(evals_df, max_date, facilities_df):
    
    '''
    Inputs:
        facilities_df: a dataframe with all evaluations information
        max_date: the maximum date in our training/testing set

    Calculates:
        Number of inspections
        Days since last inspection

        for all locations in a zip/state

        within date ranges (date2 before date1)
        before date1
    '''
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    facilities_with_features_df, evals_df = combine(evals_df, facilities_df)
    
    for group in [ids, zips, states]:
        sums = evals_df.groupby(group) \
            .size().reset_index()
        last = evals_df.groupby(group) \
            [date].max() \
            .apply(lambda x: (max_date - pd.to_datetime(x)).days) \
            .reset_index().rename(columns={date: 'last' + group})
        for gb_bool in [(sums, False), (last, True)]:
            gb, bool_val = gb_bool
            facilities_with_features_df = \
                pd.merge(facilities_with_features_df, gb, on=group, how='left')\
                .rename(columns={0: "sum_eval" + group})
            if bool_val:
                to_fill = 'last' + group
                facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill] \
                    .fillna(value=-1)
                facilities_with_features_df[to_fill] = facilities_with_features_df[to_fill]\
                    .apply(lambda x: x if x >= 0 else 0)
    

    return facilities_with_features_df.drop(columns=[zips, states])

def type_waste(facilities_df):
    '''
    calculates:
        dummy variable for waste code/ code owner/ naics code
            for a single facility
        calculates all facilities with waste code/ code owner/ naics code
            in a zip/state

    Generates features based on the type of waste created
    '''
    ids_waste = 'EPA Handler ID'
    ids_naics = 'ID_NUMBER'
    waste_codes = 'Hazardous Waste Code'
    code_owner = 'Hazardous Waste Code Owner'
    naics = 'NAICS_CODE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'

    print("Importing")
    naics_df = pd.read_csv('RCRA_NAICS.csv')
    waste_codes_df = pd.read_csv('Biennial_Report_GM_Waste_Code.csv')

    print("Merging")
    naics_df = pd.merge(\
        naics_df[[ids_naics, naics]],\
        facilities_df[[ids_naics, zips, states]],
        on=ids_naics, how='left')

    facilities_with_features_df = pd.merge(\
        naics_df[[ids_naics, naics, zips, states]],\
        waste_codes_df[[ids_waste, waste_codes, code_owner]],\
        left_on=ids_naics, right_on=ids_waste,\
        how='left')

    print("Looping")
    for col in [waste_codes, code_owner, naics]:
        ser = facilities_with_features_df[col]
        val_unique = ser.unique()
        for val in val_unique:
            print(col + " : " + str(val))
            new_col = col + str(val)
            facilities_with_features_df[new_col] = facilities_with_features_df[waste_codes]\
                .apply(lambda x: 1 if x == val else 0)
            for group in [zips, states]:
                to_merge = facilities_with_features_df.groupby(group)[new_col].sum()\
                    .reset_index()\
                    .rename(columns={new_col:new_col+group})
                facilities_with_features_df = pd.merge(facilities_with_features_df, to_merge,\
                    on=group,how='left')
        
    return facilities_with_features_df.drop(columns=[waste_codes, code_owner, zips, naics, states, ids_waste])


def snc_info(facilities_df, snc_df):
    """Adds columns for info about SNC status."""
    # TODO: Need to add ANY SNC in time period

    # year = facilities_df['YEAR_EVALUATED']
    # filter snc for just that facility,

    # create snc year column and filter out everything before 2011
    snc_df["YEAR"] = snc_df["YRMONTH"].str[:4].astype(int)
    snc_df = snc_df[snc_df["YEAR"] >= 2011]

    # filter out everything before 2011

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

    return facilities_df


def go():
    
    ids = 'ID_NUMBER'
    date = 'DATE_VIOLATION_DETERMINED'
    actual = 'ACTUAL_RTC_DATE'
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    eval_year = 'YEAR_EVALUATED'
    comp_year = 'YEAR_IN_COMPLIANCE'
    merge_date = 'DATE_TO_MERGE'
    evals_date = 'EVALUATION_START_DATE'

    violations_df = pd.read_csv('RCRA_VIOLATIONS.csv')
    #violations_df = cleaners.clean_and_converttodatetime_slashes(violations_df, date, datetime.datetime(2000,1,1,0,0))
    facilities_df = pd.read_csv('RCRA_FACILITIES.csv')
    evaluations_df = pd.read_csv('RCRA_EVALUATIONS.csv')
    snc_df = pd.read_csv('RCRA_VIOSNC_HISTORY.csv')
    for df_col in [(violations_df, date), (violations_df, actual),\
        (violations_df, scheduled), (evaluations_df, evals_date)]:
        
        df, col = df_col
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')

    violations_df[eval_year] = violations_df[date].apply(lambda x: x.year)
    violations_df[comp_year] = violations_df[actual].apply(lambda x: x.year).fillna(0)
    violations_df[comp_year] = np.where(violations_df[comp_year]==0,violations_df[eval_year] + 1,\
        violations_df[comp_year])
    evaluations_df[eval_year] = evaluations_df[evals_date].apply(lambda x: x.year)

    has_vios_df, years = has_violation(facilities_df, violations_df)

    # print("Adding snc variables")
    # with_snc_df = snc_info(facilities_df, snc_df)
    # has_vios_df = pd.merge(has_vios_df, with_snc_df[[ids, "SNC_Count", "More_Recent_SNC_Yes"]], on=ids, how="left")
    # print(has_vios_df.head())

    with_lqgs = flag_lqg(facilities_df)
    has_vios_df = pd.merge(has_vios_df, with_lqgs[[ids, "IsLQG", "IsTSDF"]], on=ids, how="left")
    num_facs = num_facilities(facilities_df)
    has_vios_df = pd.merge(has_vios_df, num_facs[[ids, "NumInMyState","NumInMyZIP"]], on=ids, how="left")
    
    """
    !!!WE NEED TO GET THIS WORKING OR SCRAP IT!!!
    print("Waste Codes")
    facs_waste = type_waste(facilities_df)
    has_vios_df = pd.merge(has_vios_df, num_facs, on=ids, how="left")

    return has_vios_df
    """

    has_vios_df = has_vios_df.drop_duplicates(subset=[ids, eval_year])

    late_early = pd.DataFrame()
    prev_evals = pd.DataFrame()
    for y in years:
        vio_filt = violations_df[violations_df[comp_year] == (y - 1)]
        eval_filt = evaluations_df[evaluations_df[eval_year] == (y - 1)]
        max_date = datetime.datetime(y, 1, 1, 0, 0)
        vio_filt = time_late_early(vio_filt, max_date, facilities_df)
        eval_filt = num_inspections(eval_filt, max_date, facilities_df)
        for filt in [vio_filt, eval_filt]:
            filt[merge_date] = y
        late_early = pd.concat([late_early, vio_filt], ignore_index=True)
        prev_evals = pd.concat([prev_evals, eval_filt], ignore_index=True)

    has_vios_df = pd.merge(has_vios_df,\
        late_early, left_on=[ids, eval_year],\
        right_on=[ids, merge_date], how="left").drop(columns=merge_date)
    has_vios_df = pd.merge(has_vios_df, prev_evals, left_on=[ids, eval_year],\
        right_on=[ids, merge_date], how="left").drop(columns=merge_date)
    
    for col in list(has_vios_df.columns):
        if col.startswith('late') or col.startswith('early') or col.startswith('sum'):
            has_vios_df[col] = has_vios_df[col].fillna(value=0)
        elif col.startswith('last'):
            has_vios_df[col] = has_vios_df[col].fillna(value=-1)

    # print("Writing to csv")
    # has_vios_df.to_csv("has_violations_features.csv")
    
    return has_vios_df

if __name__ == "__main__":
    go()
