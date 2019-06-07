'''
Bringing in new data
'''

import pandas as pd
import numpy as np
import csv

def import_data():
    '''
    '''
    col_list = ['EPA Handler ID','Activity Location','Source Type','Sequence Number','Closure Type','Expected Closure Date','New Closure Date','Date Closed','Closed In Compliance'
    ]
    increments = [13,15,16,22,23,31,39,47,48]
    csv_filename = "HD_LQG_CLOSURE.csv"
    with open(csv_filename, 'w') as to_put:
        name = "HD_LQG_CLOSURE_0.txt"
        with open(name) as orig:
            our_writer = csv.writer(to_put, delimiter=",")        
            our_writer.writerow(col_list)
            for line in orig:
                this_lst = []
                y = 0
                for x in increments:
                    this_lst.append(line[y:x])
                    y = x
                our_writer.writerow(this_lst)

def automate():
    '''
    '''
    name = "BR_REPORTING_"
    years = ["2001_", "2003_", "2005_", "2007_", "2009_",
        "2011_", "2013_", "2015_", "2017_"]
    num_lst = ["0","1"]
    col_list = ['EPA Handler ID','Activity Location','Source Type','Handler Sequence Number','Hazardous Waste Stream Page Number','Sub-page Sequence Number','BR Form','Management Location','Reporting Cycle Year','State','State Name','Region','Handler Name','Location Street Number','Location Street 1','Location Street 2','Location City','Location State','Location Zip Code','County Code','County Name','State District','Generator ID Included in NBR','Generated Waste Included in NBR','Management Facility ID Included in NBR','Managed Waste Included in NBR','Shipper ID Included in NBR','Shipped Waste Included in NBR','Receiver ID Included in NBR','Received Waste Included in NBR','Waste Description','Primary NAICS Code','Waste Source Code','Waste Form Code','Management Method Code','Federal Waste Indicator','Wastewater Characteristic Flag','Waste Generation (in tons)','Quantity Treated, Disposed, or Recycled On-site (in tons)','Total Quantity Shipped Off-site (in tons)','Quantity Received (in tons)','EPA ID Number of Facility to Which Waste was Shipped','Waste Received - State Code','Waste Received - State Name','Waste Received - EPA Region','EPA ID Number of Handler Which Shipped the Waste','Waste Shipped - State Code','Waste Shipped - State Name','Waste Shipped - EPA Region','Waste Minimization Code','Waste Code Group','Generator Status (Calculated)','Acute / Non-Acute Status','Waste Generation Activity','Priority Chemical','Management Category','Waste Property','Last Change','Federal Waste Codes'
        ]
    increments = [13,15,16,22,27,32,34,41,45,47,77,79,159,171,201,231,256,258,272,277,304,314,315,316,317,318,319,320,321,322,562,568,571,575,579,580,581,600,619,638,657,669,671,701,703,715,717,747,749,750,759,760,761,762,763,798,799,807,4807
        ]
    for y in years:
        base_name = name + y
        csv_filename = base_name + ".csv"
        import_data(col_list, base_name, csv_filename, num_lst, increments)
