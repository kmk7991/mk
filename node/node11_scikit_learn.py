import numpy as np
import matplotlib.pyplot as plt
r = np.random.RandomState(10)
x = 10 * r.rand(100)
y = 2 * x - 3 * r.rand(100)
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model)


X = x.reshape(100,1)
print(model.fit(X,y))

x_new = np.linspace(-1, 11, 100)
X_new = x_new.reshape(100,1)
y_new = model.predict(X_new)

from sklearn.metrics import mean_squared_error

error = error = np.sqrt(mean_squared_error(y,y_new))

print(error)

plt.scatter(x, y, label='input data')
plt.plot(X_new, y_new, color='red', label='regression line')

plt.show()