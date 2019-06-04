'''
Takes in violations db and returns whether or not a facility
has had a violation within the time frame
'''
import pandas as pd
import numpy as np


def has_violation(facilities_df, violations_df, start_year=2011, end_year=2018):
    '''
    facilities: list of facilities we want to know whether or not they have had a violation
    start_year: the first year we want to know about
    end_year: the last year we want to know about
    '''
    date = 'DATE_VIOLATION_DETERMINED'
    fac_id = 'ID_NUMBER'
    eval_year = 'YEAR_EVALUATED'
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
    # return facs_by_year

    violations_df[['M','D','Y']] = violations_df[date]\
        .str.split('/', expand=True)
    violations_df[eval_year] = violations_df['Y'].astype(int)


    violations_df['HasViolation'] = 1
    facs_by_year = pd.merge(facs_by_year, violations_df[['HasViolation',fac_id, eval_year]], left_on=[fac_id, eval_year], right_on=[fac_id, eval_year], how='left')
    facs_by_year['HasViolation'].fillna(0, inplace=True)

    # facs_by_year.to_csv("please_work.csv")


    return facs_by_year, years


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

def go():
    ids = 'ID_NUMBER'
    violations_df = pd.read_csv('RCRA_VIOLATIONS.csv')
    facilities_df = pd.read_csv('RCRA_FACILITIES.csv')
    has_vios_df, years = has_violation(facilities_df, violations_df)
    with_lqgs = flag_lqg(facilities_df)
    has_vios_df = pd.merge(has_vios_df, with_lqgs[[ids, "IsLQG", "IsTSDF"]], on=ids, how="left")
    num_facs = num_facilities(facilities_df)
    has_vios_df = pd.merge(has_vios_df, num_facs[[ids, "NumInMyState","NumInMyZIP"]], on=ids, how="left")


    return has_vios_df

if __name__ == "__main__":
    go()
