'''
The Features Esther Edith is creating
'''

import pandas as pd

def time_late(date1, date2, df_all_data):
    '''
    Filter by zip codes

    Filter by all facility ids
    '''
    #You can change these if need be
    #I am assuming all of these will be in the df passed here
    ids = 'ID_NUMBER'
    date_to_split = 'ACTUAL_RTC_DATE'
    actual = date_to_split
    scheduled = 'SCHEDULED_COMPLIANCE_DATE'
    zips = 'ZIP_CODE'
    states = 'STATE_CODE'
    loc = 'ACTIVITY_LOCATION'
    diff = 'difference'
    early = 'early'
    late = 'late'

    df_all_data[diff] = df_all_data[actual] - df_all_data[scheduled]
    df_all_data[diff] = df_all_data[diff]\
        .apply(lambda x: x.days)
    df_all_data[early] = df_all_data[diff]\
        .apply(lambda x: 0 if x > 0 else 1)
    df_all_data[late] = df_all_data[diff]\
        .apply(lambda x: 0 if x < 0 else 1)

    filt_between =\
        (df_all_data[date_to_split] <= date1) &\
        (df_all_data[date_to_split] >= date2)
    filt_before = (df_all_data[date_to_split] <= date1)
    
    df_between = df_all_data[filt_between]
    df_before = df_all_data[filt_before]

    for col in [early, late]:
        for group in [ids, zips, states, loc]:
            for db in [filt_between, filt_before]:
                filt = db[col] == 1
                our_db = db[filt]
                avg = our_db.groupby(group)
                    [diff].mean().reset_index().rename(\
                    columns={diff:group+col+"avg"})
                sums = our_db.groupby(group)\
                    [diff].sum().reset_index().rename(\
                    columns={diff:group+col+"sum"})
                count = our_db.groupby(group)\
                    [col].sum().reset_index().rename(\
                    columns={diff:group+col+"count"})

                for gb in [avg, sums, count]:
                    df_all_data = pd.merge(df_all_data, gb,\
                        on=group, how='left')

    return df_all_data

def num_inspections(yr1, yr2=None):
    '''
    Generates features based on the number of inspections
    yr1: we want evaluations before this
    yr2: if we want to filter dates between two dates
    '''
    csv_name = 'RCRA_EVALUATIONS.csv'
    date = 'EVALUATION_START_DATE'
    ids = 'ID_NUMBER'
    m = 'M'
    d = 'D'
    y = 'Y'

    df_ins = pd.read_csv(csv_name, usecols=[0,6])
    df_ins[[m,d,y]] = df_ins[date].str.split('/',expand=True)
    df_ins[y] = pd.to_numeric(df_ins['Y'], downcast='integer')
    if yr2 is not None:
        filt = \
            (df_ins[y] <= yr1)&\
            (df_ins[y] >= yr2)
    else:
        filt = df_ins[y] <= yr1
    df_ins = df_ins[filt]
    return df_ins.groupby(by=ids).size()

def corrective_event():
    '''
    Generates features based on a corrective action event
    
    I NEED TO RE-DOWNLOAD THIS DATA
    '''
    csv_name = 'Corrective_Action_Event.csv'
    df_ca = pd.read_csv(csv_name, usecols=[0,5])
    return df_ca

def type_waste(df_all_data):
    '''
    Generates features based on the type of waste created
    
    I NEED TO FIGURE OUT HOW TO HANDLE THE SIZE OF THIS FILE
    '''
    csv_name = 'Biennial_Report_GM_Waste_Code.csv'
    df_wc = pd.read_csv(csv_name, header=[0,6])
    waste_codes = ''
    zips = 'ZIP_CODE'
    zips_unique = df_all_data[zips].unique()
    ser = df_all_data[waste_codes]
    val_unique = ser.unique()
    
    zips_info = []
    
    for val in val_unique:
        new_col = 'waste code: ' + val
        df_all_data[new_col] = df_all_data[waste_codes]\
            .apply(lambda x: 1 if x == val else 0)
        zips_info.append(df_all_data.groupby([zips])[new_col].sum())
        
    return df_wc