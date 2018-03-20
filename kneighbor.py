# code for predicting using k - neighbor algorithm
import numpy as np

def sort_points_and_return(points, k, test_point):

    count_one = 0
    count_zero = 0

    point = np.zeros((len(points), 2))
    for i in range(len(points)):
        point[i, 0] = (points[i, 0] - test_point) ** 2
        point[i, 1] = i

    #now we'll sort this resulting point array
    for i in range(len(points)):
        min = i;
        for j in range(i + 1, len(points)):
            if point[j, 0] < point[min, 0]:
                min = j

        point[i], point[min] = point[min], point[i]

    for i in range(k):
        x = int(point[5, 1])
        if points[x , 1] == 1:
            count_one = count_one + 1
        else:
            count_zero = count_zero + 1

    if count_one > count_zero:
        return 0
    else:
        return 1


def predict(points, k, test):
    target = np.zeros(len(test))    # stores the result for the test points

    for i in range(len(test)):
        target[i] = sort_points_and_return(points, k, test[i])
    return target


def run():
    points = np.genfromtxt('dataForKN.csv', delimiter=',')
    k = 7
    test = np.array([32.4527, 53.4203, 61.5803, 47.3963, 59.8787, 55.14841, 52.21169, 39.56669, 48.10569, 52.5500])
    target = predict(np.array(points), k, test)
    print("target : ", target)


if __name__ == '__main__':
    run()
