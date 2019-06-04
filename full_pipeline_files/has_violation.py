'''
Takes in violations db and returns whether or not a facility
has had a violation within the time frame
'''
import pandas as pd

violation_df = pd.read_csv('RCRA_VIOLATIONS.csv')
facilities_df = pd.read_csv('RCRA_FACILITIES.csv')

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
        for y in year:
            id_lst.append(fid)
            year_lst.append(y)

    data = {fac_id: id_lst, eval_year: year_lst}
    facs_by_year = pd.dataframe(data=data)
    return facs_by_year

    violations_df[['M','D','Y']] = violations_df[date]\
        .str.split('/', expand=True)
    violations_df['Y'] = violations_df['Y'].astype(int)
    filt = (violations_df['Y'] >= start_year and\
        violations_df['Y'] <= end_year)
    filt_vio = violations_df[filt]
    

    
