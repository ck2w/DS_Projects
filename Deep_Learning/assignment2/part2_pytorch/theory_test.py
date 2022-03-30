
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.where(x > 0, x, 0)

def compute(x):
    h1 = W1 * x + b1
    r1 = relu(h1)
    h2 = W2 @ r1 + b2
    r2 = relu(h2)
    h = W3 @ r2 + b3
    return h

W1 = np.array([[1.5], [0.5]])
b1 = np.array([[0], [1]])
W2 = np.array([[1, 2], [2, 1]])
b2 = np.array([[0], [1]])
W3 = np.array([1, 1])
b3 = -1


x_lst = np.linspace(-3, 3, 30)
y = []
for x in x_lst:
    y.append(compute(np.array([x])))

plt.plot(x_lst, y)
plt.show()