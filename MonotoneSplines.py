#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvx


def contained_interval(x, knots_augmented):
    """Return the index of the interval [knots_augmented[i - 1], knots_augmented[i]] that contains the point x"""
    for i in range(len(knots_augmented)):
        if x < knots_augmented[i]:
            return i - 1
    return len(knots_augmented) - 1


def Mspline(x, k, i, knots_augmented):
    """Returns the value of the M-spline at the point x for the given parameters k and i"""
    if k == 1:
        if knots_augmented[i] == knots_augmented[i + 1]:
            return 0.
        if knots_augmented[i] <= x and x < knots_augmented[i + 1] and knots_augmented[i] != knots_augmented[i + 1]:
            return 1/(knots_augmented[i + 1] - knots_augmented[i])
        else:
            return 0.
    else:
        if knots_augmented[i] == knots_augmented[i + k]:
            return 0.
        return k * (
            (x - knots_augmented[i]) * Mspline(x, k - 1, i, knots_augmented)
            + (knots_augmented[i + k] - x) * Mspline(x, k - 1, i + 1, knots_augmented)
        ) / (
            (k - 1) * (knots_augmented[i + k] - knots_augmented[i])
        )


def Ispline(x, k, i, knots_augmented):
    """Returns the value of the I-spline at the point x for the given parameters k and i"""
    result = 0.
    j = contained_interval(x, knots_augmented)
    if j < i:
        return 0.
    elif j - k + 1 <= i and i <= j:
        for i_ in range(i, j + 1):
            result += (knots_augmented[i_ + k + 1] - knots_augmented[i_]) * Mspline(x, k + 1, i_, knots_augmented) / (k + 1)
        return result
    else:
        return 1.


def show_plots():
    m = 5   # Number of interior breakpoints
    a = 0.0   # Start of interval
    b = 1.0   # End of interval
    xs = np.linspace(a, b, 200)
    knots = np.linspace(a, b, m + 2)

    # Degree k = 1
    k = 1
    knots_augmented = np.concatenate([np.repeat(a, k - 1), knots, np.repeat(b, k - 1)])
    for i in range(m + k):
        plt.plot(xs, list(map(lambda x: Mspline(x, 1, i, knots_augmented), xs)))
    plt.show()

    # Degree k = 2
    k = 2
    knots_augmented = np.concatenate([np.repeat(a, k - 1), knots, np.repeat(b, k - 1)])
    for i in range(m + k):
        plt.plot(xs, list(map(lambda x: Mspline(x, 2, i, knots_augmented), xs)))
    plt.show()
    for i in range(m + k - 1):
        plt.plot(xs, list(map(lambda x: Ispline(x, 1, i + 1, knots_augmented), xs)))
    plt.show()

    # Degree k = 3
    k = 3
    knots_augmented = np.concatenate([np.repeat(a, k - 1), knots, np.repeat(b, k - 1)])
    for i in range(m + k):
        plt.plot(xs, list(map(lambda x: Mspline(x, 3, i, knots_augmented), xs)))

    for i in range(m + k - 1):
        plt.plot(xs, list(map(lambda x: Ispline(x, 2, i + 1, knots_augmented), xs)))

    # Degree k = 4
    k = 4
    knots_augmented = np.concatenate([np.repeat(a, k - 1), knots, np.repeat(b, k - 1)])
    for i in range(m + k):
        plt.plot(xs, list(map(lambda x: Mspline(x, 4, i, knots_augmented), xs)))

    for i in range(m + k - 1):
        plt.plot(xs, list(map(lambda x: Ispline(x, 3, i + 1, knots_augmented), xs)))


def phi_vectorized(xs, m, k, knots_augmented):
    num_points = len(xs)
    column_list = [np.ones((num_points, 1))]
    for i in range(m + k - 1):
        print(f'\rProcessing feature {i + 1} of {m + k - 1}', end='')

        def Ispline_wrapper(x):
            return Ispline(x, k - 1, i + 1, knots_augmented)

        Ispline_vectorized = np.vectorize(Ispline_wrapper)
        column_list.append(Ispline_vectorized(xs).reshape(-1, 1))
    print('\n', end='')  # Print \n after using \r in loop
    Phi = np.concatenate(column_list, axis=1)
    return Phi


def fit_model_decreasing(x, y):
    # Define the objective function
    Phi = phi_vectorized(x)
    D = Phi.shape[1]
    P = cvx.matrix(np.matmul(Phi.transpose(), Phi))
    q = cvx.matrix(-np.matmul(Phi.transpose(), y))

    # Define the constraints
    G_array = np.identity(D)  # Positive 1 for decreasing fit
    G_array = G_array[1:, ]  # Don't constrain the bias term
    h_array = np.zeros(len(G_array))
    G = cvx.matrix(G_array)
    h = cvx.matrix(h_array)
    solution = cvx.solvers.qp(P, q, G, h)
    w = np.array(solution['x'])

    return w


def fit_model_increasing(x, y):
    # Define the objective function
    Phi = phi_vectorized(x)
    D = Phi.shape[1]
    P = cvx.matrix(np.matmul(Phi.transpose(), Phi))
    q = cvx.matrix(-np.matmul(Phi.transpose(), y))

    # Define the constraints
    G_array = - np.identity(D)  # Negative 1 for increasing fit
    G_array = G_array[1:, ]  # Don't constrain the bias term
    h_array = np.zeros(len(G_array))
    G = cvx.matrix(G_array)
    h = cvx.matrix(h_array)
    solution = cvx.solvers.qp(P, q, G, h)
    w = np.array(solution['x'])

    return w


def predict(xs, w):
    Phi = phi_vectorized(xs)
    return np.matmul(Phi, w)


def test():
    def build_increasing_function(z=40000):
        """Example true signal function for generating data"""
        a = 1 / (z**2)

        def f(d):
            return a * d**2

        return f

    def build_decreasing_function(z=20_000):
        """Example true signal function for generating data"""
        a = -1 / (z**2)

        def f(d):
            if d >= 0 and d <= z:
                return a * d**2 + 1
            else:
                return 0.0
        return f

    rng = np.random.default_rng(2023)
    x_max = 40_000
    n = 1_000_000
    x = x_max * rng.random(n)
    true_function = build_increasing_function()
    probabilities = np.array(list(map(true_function, x)))
    iterates = rng.random(n)
    y = (iterates <= probabilities).astype(np.float64)

    m = 20  # Number of interior breakpoints
    a = 0.0
    b = x_max
    x_plt = np.linspace(a, b, 1000)
    knots = np.linspace(a, b, m + 2)
    k = 4  # For order 3 I-Splines
    knots_augmented = np.concatenate([np.repeat(a, k - 1), knots, np.repeat(b, k - 1)])

    plt.plot(x_plt, list(map(true_function, x_plt)))
    plt.show()

    for i in range(m + k - 1):
        y_plt = np.array(list(map(lambda x: Ispline(x, k - 1, i + 1, knots_augmented), x_plt)))
        plt.plot(x_plt, y_plt)
    plt.show()




    ys = predict(xs, w)

    w = fit_model_decreasing(x, y)
    plt.plot(x_plt, list(map(signal_function_true, x_plt)))
    plt.plot(x_plt, list(map(lambda x: predict(x, w), x_plt)))
    plt.show()

    plt.plot(x_plt, list(map(signal_function_true, x_plt)))

if __name__ == '__main__':
    test()