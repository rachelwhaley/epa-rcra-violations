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
