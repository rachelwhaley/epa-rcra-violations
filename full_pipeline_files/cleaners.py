import pandas as pd

'''
cleaning functions for the epa pipeline
'''

def clean_converttodatetime_dashes(EVALS, date_col):
    EVALS[date_col] = pd.to_datetime(EVALS.EVALUATION_START_DATE,
                                    format='%m/%d/%Y', errors='coerce')
    oor_decade_markers = {'1920-01-01': 100, '1930-01-01': 60, '1940-01-01': 50,
                         '1960-01-01': 30, '1970-01-01': 20, '1980-01-01': 10}
    for k, v in oor_decade_markers.items():
        marker = pd.to_datetime(k)
        EVALS[date_col] = \
        EVALS[date_col].apply(lambda x: x + pd.DateOffset(years=v)
                                            if x < marker else x)

def clean_and_convert_to_datetime_slashes(df, stand_form_col):
    '''
    looks for out of range dates and brings them to range
    '''
    oor_dict = {'19': 2000, '1919': 100, '1943': 50, '1971': 20, '1974': 20,
               '1979': 18}
    df[['M','D','Y']] = df[stand_form_col].str.split('/', expand=True)
    df['Y'] = df.Y.astype(int)

    
    for key, value in oor_dict.items():
        k = int(key)
        if k < 1943:
            df['Y'] = df['Y'].apply(lambda x: x + value if x <= k else x)
        else:
            df['Y'] = df['Y'].apply(lambda x: x + value if x == k else x)
    df['Y'] = df.Y.astype('str')
    df[stand_form_col] = df[['M','D', 'Y']].apply(lambda x: '/'.join(x), axis = 1)
    df[stand_form_col] = df.DATE_VIOLATION_DETERMINED.astype('datetime64')
    df.drop(['M','D','Y'], axis=1, inplace=True)

def yr_month_to_datetime(df, year_month_col):
    df[year_month_col] = pd.to_datetime(df[year_month_col], format='%Y%m',
                                        errors='coerce')
