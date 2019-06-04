'''
Takes in violations db and returns whether or not a facility
has had a violation within the time frame
'''
import pandas as pd
import numpy as np


def has_violation(facilities_df, violations_df, start_year, end_year):
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

    print(facs_by_year.head())

    violations_df['HasViolation'] = 1
    facs_by_year = pd.merge(facs_by_year, violations_df[['HasViolation',fac_id, eval_year]], left_on=[fac_id, eval_year], right_on=[fac_id, eval_year], how='left')
    facs_by_year['HasViolation'].fillna(0, inplace=True)

    # facs_by_year.to_csv("please_work.csv")

    print(facs_by_year["HasViolation"].describe())


    return facs_by_year


def flag_lqg(facilities_df):
    """Adds a column to the data frame that is 1 if the facility is an LQG."""
    # 3 is the code for LQG according to the data documentation at
    #   https://echo.epa.gov/tools/data-downloads/rcrainfo-download-summary#download_files
    facilities_df["IsLQG"] = pd.Series(np.where(facilities_df.FED_WASTE_GENERATOR.str.contains("3"), 1, 0), \
                                       facilities_df.index)
    facilities_df["IsTSDF"] = pd.Series(np.where(facilities_df.HREPORT_UNIVERSE_RECORD.str.contains("TSDF"), 1, 0), \
                                        facilities_df.index)

    return facilities_df

def go():
    print("Uploading")
    violations_df = pd.read_csv('RCRA_VIOLATIONS.csv')
    facilities_df = pd.read_csv('RCRA_FACILITIES.csv')
    print("Uploaded")
    print("creating base df")
    has_vios_df = has_violation(facilities_df, violations_df, 2011, 2018)
    print("with lqgs")
    with_lqgs = flag_lqg(facilities_df)
    has_vios_df = pd.merge(has_vios_df, with_lqgs[['ID_NUMBER', "IsLQG", "IsTSDF"]], on="ID_NUMBER", how="left")

    print(has_vios_df["IsLQG"].describe())
    print(has_vios_df['IsTSDF'].describe())

    #has_vios_df.to_csv("has_vios.csv")

    print(has_vios_df.head())

    return has_vios_df

if __name__ == "__main__":
    go()
