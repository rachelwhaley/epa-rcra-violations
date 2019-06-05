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
    had = 'ViolationLastYear'
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
    
    violations_df[next_year] = violations_df[eval_year] + 1
    violations_df[had] = 1
    facs_by_year = pd.merge(facs_by_year, violations_df[[had,fac_id, next_year]],\
        left_on=[fac_id, eval_year], right_on=[fac_id, next_year], how='left')

    facs_by_year = pd.merge(facs_by_year, facilities_with_features_df, on=fac_id, how='left')

    for y in years:
        vios_this_yr = violations_df[violations_df[next_year] == y]
        max_date = datetime.datetime(y, 1, 1, 0, 0)
        print(vios_this_yr.head())
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
                gb[next_year] = y
                facs_by_year = pd.merge(facs_by_year, gb,\
                    left_on=[group, eval_year], right_on=[group, next_year], how='left')
                print("Merged " + group)

    for col in [had, has_vio]:
        facs_by_year[col].fillna(0, inplace=True)

    return facs_by_year.drop_duplicates(subset=[fac_id, eval_year]).drop(columns=next_year), years


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
    return facilities_with_features_df, more_info_df

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

def go():
    ids = 'ID_NUMBER'
    date = 'DATE_VIOLATION_DETERMINED'
    actual = 'ACTUAL_RTC_DATE'
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    eval_year = 'YEAR_EVALUATED'
    comp_year = 'YEAR_IN_COMPLIANCE'
    merge_date = 'DATE_TO_MERGE'

    violations_df = pd.read_csv('RCRA_VIOLATIONS.csv')
    #violations_df = cleaners.clean_and_converttodatetime_slashes(violations_df, date, datetime.datetime(2000,1,1,0,0))
    facilities_df = pd.read_csv('RCRA_FACILITIES.csv')
    for col in [date, actual, scheduled]:
        violations_df[col] = pd.to_datetime(violations_df[col], format='%m/%d/%Y', errors='coerce')
    
    violations_df[eval_year] = violations_df[date].apply(lambda x: x.year)
    violations_df[comp_year] = violations_df[actual].apply(lambda x: x.year).fillna(0)
    violations_df[comp_year] = np.where(violations_df[comp_year]==0,violations_df[eval_year] + 1,\
        violations_df[comp_year])

    has_vios_df, years = has_violation(facilities_df, violations_df)
    return has_vios_df
    with_lqgs = flag_lqg(facilities_df)
    has_vios_df = pd.merge(has_vios_df, with_lqgs[[ids, "IsLQG", "IsTSDF"]], on=ids, how="left")
    num_facs = num_facilities(facilities_df)
    has_vios_df = pd.merge(has_vios_df, num_facs[[ids, "NumInMyState","NumInMyZIP"]], on=ids, how="left")

    late_early = pd.DataFrame()
    for y in years:
        print(y)
        filt = violations_df[comp_year] == (y - 1)
        vio_filt = violations_df[filt]
        max_date = datetime.datetime(y, 1, 1, 0, 0)
        vio_filt = time_late_early(vio_filt, max_date, facilities_df)
        print(vio_filt.head())
        vio_filt[merge_date] = y
        if late_early.empty:
            late_early = vio_filt
        else:
            late_early = pd.concat([late_early, vio_filt], ignore_index=True)

    has_vios_df = pd.merge(has_vios_df, vio_filt, left_on=[ids, eval_year],\
        right_on=[ids, merge_date], how="left").drop(columns=merge_date)
    for col in list(has_vios_df.columns):
        if col.startswith('late') or col.startswith('early'):
            has_vios_df[col] = has_vios_df[col].fillna(value=0)
        elif col.startswith('last'):
            has_vios_df[col] = has_vios_df[col].fillna(value=float(-1))

    return has_vios_df

if __name__ == "__main__":
    go()
