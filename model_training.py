import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('day.csv', parse_dates=['dteday'])
data.head()

# Create a feature day of year to include month and day information
data['dayofyear']=(data['dteday']-data['dteday'].apply(lambda x: date(x.year,1,1)).astype('datetime64[ns]')).apply(lambda x: x.days)
offset = int(len(data)*0.8)

# Create variables and remove unused columns
X = np.array(data.drop(['dteday','mnth','casual','registered','cnt'],axis=1))
Y = np.array(data['cnt'])

# Split train and test
X_train, X_test = X[:offset], X[offset:]
Y_train, Y_test = Y[:offset], Y[offset:]

# Set up and learn a RF regessor
RF = RandomForestRegressor()
RF.fit(X_train,Y_train)

# Mittlere quadratische Abweichung (MSE)
print(mean_squared_error(Y_test,RF.predict(X_test)))

# Predicting Median
print(mean_squared_error(Y_test,np.median(Y_train)*np.ones(len(Y_test))))

# Save Model as a pickle Object
with open("modelfile.pickle", 'wb') as handle:
    pickle.dump(RF, handle, protocol=pickle.HIGHEST_PROTOCOL)
