#import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('house_data.csv')

Xtrain, Xtest, ytrain, ytest = train_test_split(df[['bedrooms','bathrooms','sqft_living']],df[['price']])

lrg = LinearRegression()

lrg.fit(Xtrain,ytrain)

#arr = np.array([12,12,5454])
#arr = arr.astype(np.float64)
#pred = lrg.predict([arr])
#print(pred)

with open('lrg.pkl', 'wb') as f:  # open a text file
    pickle.dump(lrg, f) # serialize the list


