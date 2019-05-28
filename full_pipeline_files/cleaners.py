import pandas as pd

def clean_evals(EVALS):
    wrong_date_1 = pd.to_datetime('1920-01-01')
    EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=100)
                                        if x < wrong_date_1 else x)
    wrong_date_2 = pd.to_datetime('1930-01-01')
    EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=60)
                                        if x < wrong_date_2 else x)
    wrong_date_3 = pd.to_datetime('1940-01-01')
    EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=50)
                                        if x < wrong_date_3 else x)
    wrong_date_4 = pd.to_datetime('1960-01-01')
    EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=30)
                                        if x < wrong_date_4 else x)
    wrong_date_5 = pd.to_datetime('1970-01-01')
    EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=20)
                                        if x < wrong_date_5 else x)
    wrong_date_6 = pd.to_datetime('1980-01-01')
    EVALS['EVALUATION_START_DATE'] = \
    EVALS['EVALUATION_START_DATE'].apply(lambda x: x + pd.DateOffset(years=10)
                                        if x < wrong_date_6 else x)

def clean_vios(df):
    df[['M','D','Y']] = df['DATE_VIOLATION_DETERMINED'].str.split('/', expand=True)
    df['Y'] = df.Y.astype(int)
    df['Y'] = df['Y'].apply(lambda x: x + 2000 if x <= 19 else x)
    df['Y'] = df['Y'].apply(lambda x: x + 100 if x <= 1919 else x)
    df['Y'] = df['Y'].apply(lambda x: x + 50 if x == 1943 else x)
    df['Y'] = df['Y'].apply(lambda x: x + 20 if x ==1971 else x)
    df['Y'] = df.Y.apply(lambda x: x + 20 if x == 1974 else x)
    df['Y'] = df.Y.apply(lambda x: 1997 if x == 1979 else x)
    df['Y'] = df.Y.astype('str')
    df['DATE_VIOLATION_DETERMINED'] = df[['M','D', 'Y']].apply(lambda x: '/'.join(x), axis = 1)
    df['DATE_VIOLATION_DETERMINED'] = df.DATE_VIOLATION_DETERMINED.astype('datetime64')
    df.drop(['M','D','Y'], axis=1, inplace=True)

def clean_snc(SNC):
    SNC['YRMONTH'] = pd.to_datetime(SNC['YRMONTH'], format='%Y%m', errors='coerce')
