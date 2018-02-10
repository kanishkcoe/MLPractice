import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn import datasets


def compute_error_for_given_points(b, m, points):
    totalError = 0

    for j in range(0, len(points[0])):
        for i in range(0, len(points)):
            x = points[i, j]
            y = points[i, len(points[0]) - 1]
            totalError += (y - (m[j] * x + b))**2
    return totalError / float(len(points))


def step_gradient(current_b, current_m, points, learning_rate):
    b_gradient = 0
    m_gradient = np.array(0 for i in range(len(points)))
    N = float(len(points[0]))

    for j in range(0, len(points[0])):
        for i in range(0, len(points)):
            y = points[i, len(points[0] - 1)]
            x = points[i, j]
            b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
            m_gradient[j] += -(2/N) * x * (y - ((current_m * x) + current_b))
        new_b = current_b - (learning_rate * b_gradient)
        new_m = current_m - (learning_rate * m_gradient)
    return (new_b, new_m)


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return b, m


def run():
    diabetes = datasets.load_diabetes()
    # points to be entered
    df = pd.DataFrame(diabetes.data)

    # hyperparameters
    learning_rate = 0.0001

    # linear regression formula
    initial_b = 0
    initial_m = np.array(0 for i in range(len(points)))

    num_iterations = 1000

    b, m = gradient_descent_runner(df, initial_b, initial_m, learning_rate)
    print("intercept = ", b)
    print("slopes : ")
    for i in range(10):
        print("m", i, " = ", m[i])


if __name__ == '__main__':
    run()
