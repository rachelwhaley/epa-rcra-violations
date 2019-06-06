'''
Bringing in new data
'''

import pandas as pd
import numpy as np
import csv

def import_data():
    '''
    '''
    col_list = ['EPA Handler ID','Activity Location','Source Type',
        'Handler Sequence Number','Hazardous Waste Stream GM Form Page Number',
        'Hazardous Waste Code Owner','Hazardous Waste Code']

    base_name = 'BR_GM_WASTE_CODE_0.txt'
    with open(base_name) as orig:
        with open(csv_filename, 'w') as to_put:
        our_writer = csv.writer(to_put, delimiter=",")        
        our_writer.writerow(col_list)
        for line in orig:
            this_lst = []
            y = 0
            for x in [13,15,16,22,27,29,35]:
                this_lst.append(line[y:x])
                y = x
            our_writer.writerow(this_lst)

    '''
    col_list = ['EPA Handler ID','Area Sequence Number',
        'Entire Facility Indicator','Regulated Unit Indicator / Area Name',
        'Air Release Indicator','Groundwater Release Indicator',
        'Soil Release Indicator','Surface Water Release Indicator',
        'EPA Responsible Person Owner','EPA Responsible Person',
        'State Responsible Person Owner','State Responsible Person']

    df_lst = []
    with open('CA_AREA_0.txt') as CA:
        for line in CA:
            this_lst = []
            y = 0
            for x in [13,17,18,59,60,61,62,63,65,70,72,77]:
                this_lst.append(line[y:x])
                y = x
            df_lst.append(this_lst)
    df_down = pd.DataFrame(np.array(df_lst), columns=col_list)
    df_down.to_csv("Corrective_Action_Area.csv")
    '''
