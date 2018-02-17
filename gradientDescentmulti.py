import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn import datasets

def errorFunction(b, m, points, target):
    totalMterms = m * points
    y = target

    msum = totalMterms.sum()

    error = y - msum
    return error

def compute_error_for_given_points(b, m, points):
    totalError = 0

    for j in range(0, len(points[0])):
        for i in range(0, len(points)):
            x = points[i, j]
            y = points[i, len(points[0]) - 1]
            totalError += (y - (m[j] * x + b))**2
    return totalError / float(len(points))


def step_gradient(current_b, current_m, points, target, learning_rate):
    b_gradient = 0
    m_gradient = np.zeros(len(points[0]))
    N = int(len(points))

    for i in range(0, len(points)):
        y = target[i]
        x = points[i]
        b_gradient += -(2 / N) * errorFunction(current_b, current_m, x, y)
        m_gradient = np.add(m_gradient, np.multiply((-2 / N) * errorFunction(current_b, current_m, x, y), x))
    new_b = current_b - (learning_rate * b_gradient)
    new_m = np.subtract(current_m, np.multiply(learning_rate, m_gradient))
    return (new_b, new_m)


def gradient_descent_runner(points, target, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, target, learning_rate)
    return b, m


def run():
    boston = datasets.load_boston()
    # points to be entered
    df = pd.DataFrame(boston.data)
    df = np.array(df.values)
    target = boston.target

    # hyperparameters
    learning_rate = 0.0001

    # linear regression formula
    initial_b = 0
    initial_m = np.zeros(len(df[0]))
    print("intial m = ", initial_m)

    num_iterations = 150

    b, m = gradient_descent_runner(df, target, initial_b, initial_m, learning_rate, num_iterations)
    print("intercept = ", b)
    print("slopes : ", m)


if __name__ == '__main__':
    run()
