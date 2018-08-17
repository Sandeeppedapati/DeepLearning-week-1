import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score

# Reading and Retrieving data
data = pd.read_csv('firstchallengedataset.txt',header=None)
print(data)
x_data = data[[0]]
y_data = data[[1]]

#training the model
model = linear_model.LinearRegression()
model.fit(x_data,y_data)

#visualizing the data
plt.scatter(x_data,y_data)
pred = model.predict(x_data)
plt.plot(x_data,pred)
plt.show()

#accuracy checking
w = []
l=[]
for x in np.nditer(pred) :
    w.append(float(x))
for index,row in data.iterrows() :
    l.append(row[1])
print(len(w))
print(len(l))
print("mean squared error is",mean_squared_error(y_data,pred))
print("r2_score is",r2_score(y_data,pred))

