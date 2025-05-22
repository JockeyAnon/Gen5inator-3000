from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt

def logistic(x, alpha=10, mu=1):
    expn = np.exp(-alpha * (x - mu))
    return 1 / (1 + expn)

def d_logistic_dx(x, alpha=10, mu=1):
    expn = np.exp(-alpha * (x - mu))
    return -(expn * alpha * (x - mu)) / (1 + expn)**2

def rbf(x, alpha=10, mu=1):
    lambd = logistic(x, alpha, mu)
    return lambd - lambd**2

def d_rbf_dx(x, alpha=10, mu=1):
    lambd = logistic(x, alpha, mu)
    dl_dx = d_logistic_dx(x, alpha, mu)
    return dl_dx * lambd * (np.exp(-alpha * (x - mu)) - 1)

xs = np.linspace(0.1, 2)
# Made up cell fold-change data
ys1 = logistic(xs) + np.random.normal(0, .05, len(xs)) 
ys2 = d_logistic_dx(xs) + np.random.normal(0, .02, len(xs)) 
ys3 = rbf(xs) + np.random.normal(0, .04, len(xs)) 
ys4 = d_rbf_dx(xs) + np.random.normal(0, .02, len(xs)) 

data_matrix = np.array([ys1, ys2, ys3])

def my_model(y1, y2, y3, parameters):
    """
    Uses y1, y2 and y3 to predict y4.
    """
    a, b, c, d, e, f, g = parameters
    return a + b * y1 + c * y2 + d * y3 + e * y1 * y2 + f * y1*y3 +  g * y2 * y3

def predictions_from_my_model(parameters):
    """
    Returns my model's attempt to predict the ys4 data.
    """
    predictions = [my_model(ys1[i], ys2[i], ys3[i], parameters) for i in range(len(ys1))]
    return np.array(predictions)

def error_function2minimize(parameters):
    """
    The error vector that we want to minimize the sum of squares of.
    """
    return ys4 - predictions_from_my_model(parameters)


results = least_squares(error_function2minimize, [.1, 2, 2, 2, 0.51, 0.41, 0.61], method='lm')
params = results.x
print(params)
plt.plot(xs, ys1, label='logistic', alpha=0.5)
plt.plot(xs, ys2, label='d_logistic_dx', alpha=0.5)
plt.plot(xs, ys3, label='rbf', alpha=0.5)
plt.plot(xs, ys4, '.', label='d_rbf_dx', alpha=0.5)
plt.plot(xs, [my_model(ys1[i], ys2[i], ys3[i], params) for i in range(len(xs))], label="final model predictions")
plt.legend()
plt.show()