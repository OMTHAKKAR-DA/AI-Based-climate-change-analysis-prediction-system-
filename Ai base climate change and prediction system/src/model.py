import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/climate_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

X = data.index.values.reshape(-1,1)
y = data['Temperature']

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)

plt.plot(data['Date'], y, label='Actual')
plt.plot(data['Date'], pred, label='Predicted')
plt.legend()
plt.savefig('output.png')
plt.show()
