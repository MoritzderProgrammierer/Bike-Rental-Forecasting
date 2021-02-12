import numpy as np
import pandas as pd
from datetime import date


def doTheCalculation(data):
    print(data['dayofyear'])
    data['dayofyear'] = (data['dteday'] - data['dteday'].apply(
        lambda x: date(x.year, 1, 1)).astype('datetime64[ns]')).apply(lambda x: x.days)

    X = np.array(data[['instant', 'season', 'yr', 'holiday', 'weekday', 'workingday',
                       'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'dayofyear']])
    return X
