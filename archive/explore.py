import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn import metrics#Step 1: Read the data


def read_data(filename):
    '''
    Read the data and convert it to a dataframe

    Input:
        filename(csv): data file
    Output:
        dataframe

    '''
    data =pd.read_csv(filename)
    return data
