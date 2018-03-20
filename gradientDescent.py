import numpy as np


def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b))**2
    return totalError / float(len(points))


def step_gradient(current_b, current_m, points, learning_rate):
    # gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((current_m * x) + current_b))
        m_gradient += -(2 / N) * x * (y - ((current_m * x) + current_b))
    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)
    return (new_b, new_m)


def gradient_descent_runner(points, initial_b, intial_m, learning_rate, num_iterations):
    b = initial_b
    m = intial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return b, m


def run():
    # points to be entered
    points = np.genfromtxt('data.csv', delimiter=',')
    # hyperparameters
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    intial_m = 0
    num_iterations = 1000
    print(points[1])
    b, m = gradient_descent_runner(points, initial_b, intial_m, learning_rate, num_iterations)
    print("intercept = ", b)
    print("slope = ", m)


if __name__ == '__main__':
    run()
