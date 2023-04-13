#!/usr/bin/env python
# coding: utf-8
"""Module for computing I-splines, which are either non-decreasing or non-increasing"""

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as cvx


def contained_interval(x, knots_augmented):
    """Return the index of the interval [knots_augmented[i - 1], knots_augmented[i]] that contains
    the point x

    Parameters
    ----------
    x : numeric
        The x-value to test
    knots_augmented : np.array
        The vector of augmented knots

    Returns
    -------
    int
        The index of the interval containing x
    """
    for i in range(len(knots_augmented)):
        if x < knots_augmented[i]:
            return i - 1
    return len(knots_augmented) - 1


def Mspline(x, k, i, knots_augmented):
    """Returns the value of the ith M-spline basis function at the point x for the given degree k

    Parameters
    ----------
    x : int, float
        The x-value to compute the function at
    k : int
        The degree of the spline
    i : int
        The index of the spline basis to compute
    knots_augmented : np.array
        The vector of augmented knots

    Returns
    -------
    float
        The y-value of the computed spline basis function
    """
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
    """Returns the value of the ith I-spline basis function at the point x for the given degree k

    Parameters
    ----------
    x : int, float
        The x-value to compute the function at
    k : int
        The degree of the spline (for I-splines, the polynomial degree of the basis will be k - 1)
    i : int
        The index of the spline basis to compute
    knots_augmented : np.array
        The vector of augmented knots

    Returns
    -------
    float
        The y-value of the computed spline basis function
    """
    result = 0.
    j = contained_interval(x, knots_augmented)
    if j < i:
        return 0.
    elif j - k + 1 <= i and i <= j:
        for i_ in range(i, j + 1):
            result += (
                (knots_augmented[i_ + k + 1] - knots_augmented[i_])
                * Mspline(x, k + 1, i_, knots_augmented)
                / (k + 1)
            )
        return result
    else:
        return 1.


class IntegratedSplines:
    """Integrated Spline (I-spline) object. Can fit models using either an increasing or decreasing I-spline model.
    """
    def __init__(
            self,
            xs,
            ys,
            mode,
            num_knots,
            degree=4,
            support_min=None,
            support_max=None,
            verbose=True
        ):
        """_summary_

        Parameters
        ----------
        xs : np.array
            The x-values for the model
        ys : np.array
            The y-values for the model
        mode : str
            Either in ['inc', 'i', 'increase', 'increasing'] for an increasing fit or in
            ['dec', 'd', 'decrease', 'decreasing'] for a decreasing fit
        num_knots : int
            The number of interior knots to use
        degree : int, optional
            The degree of the spline fit (for I-splines, the polynomial degree of the basis will
            be degree - 1), by default 4 for cubic splines
        support_min : numeric, optional
            Left endpoint for the spline basis, uses min(xs) if None, by default None
        support_max : numeric, optional
            Right endpoint for the spline basis, uses min(xs) if None, by default None

        Raises
        ------
        ValueError
            If mode is not set to 'increasing' or 'decreasing'
        """
        self.xs = xs
        self.ys = ys
        self.verbose = verbose

        if mode.lower() in ['inc', 'i', 'increase', 'increasing']:
            self.mode = 'increasing'
        elif mode.lower() in ['dec', 'd', 'decrease', 'decreasing']:
            self.mode = 'decreasing'
        else:
            raise ValueError('mode must be either increasing or decreasing')

        if support_min is not None:
            self.support_min = support_min
        else:
            self.support_min = np.min(xs)

        if support_max is not None:
            self.support_max = support_max
        else:
            self.support_max = np.max(xs)

        self.num_knots = num_knots
        self.degree = degree
        self.knots = np.linspace(self.support_min, self.support_max, num_knots + 2)
        self.knots_augmented = np.concatenate(
            [
                np.repeat(self.support_min, self.degree - 1),
                self.knots,
                np.repeat(self.support_max, self.degree - 1)
            ]
        )

        self.design_matrix = build_design_matrix(xs, self.num_knots, self.degree, self.knots_augmented)

        if mode == 'increasing':
            self.weights_optimal = fit_model_increasing(self.design_matrix, self.ys)
        elif mode == 'decreasing':
            self.weights_optimal = fit_model_decreasing(self.design_matrix, self.ys)

    def predict(self, xs):
        """Make predictions for a new vector of xs

        Parameters
        ----------
        xs : np.array
            x-values to predict on

        Returns
        -------
        np.array
            y-values containing predictions for the given x-values
        """
        design_matrix = build_design_matrix(
            xs,
            self.num_knots,
            self.degree,
            self.knots_augmented,
            verbose=self.verbose
        )
        return np.matmul(design_matrix, self.weights_optimal)

    def show(self):
        """Plot the fitted spline function
        """
        x_plt = np.linspace(self.support_min, self.support_max, num=3000)
        y_plt = self.predict(x_plt)
        plt.plot(x_plt, y_plt)
        plt.show()

    def show_design_matrix(self):
        """Plot the spline basis functions used in the design matrix
        """
        x_plt = np.linspace(self.support_min, self.support_max, num=3000)
        design_matrix_plt = build_design_matrix(
            x_plt,
            self.num_knots,
            self.degree,
            self.knots_augmented
        )
        for i in range(design_matrix_plt.shape[1]):
            plt.plot(x_plt, design_matrix_plt[:, i])
        plt.show()


def build_design_matrix(xs, m, k, knots_augmented, verbose=False):
    """Builds the design matrix Phi, where the ith column is the value of the ith basis spline
    computed on each x-value in xs

    Parameters
    ----------
    xs : np.array
        x-values to predict on
    m : int
        Number of (interior) knots. The total number of augmented (interior + exterior) knots will
        be m + 2
    k : int
        Degree of the monotone spline. For integrated splines, the actual polynomial degree will be
        k - 1
    knots_augmented : np.array
        The augmented vector of knots to use for the spline fit
    verbose : bool, optional
        If True, will print progress for which features have been computed, by default False

    Returns
    -------
    np.array
        The design matrix Phi
    """
    num_points = len(xs)
    column_list = [np.ones((num_points, 1))]
    for i in range(m + k - 1):
        if verbose:
            print(f'\rProcessing feature {i + 1} of {m + k - 1}', end='')

        def Ispline_wrapper(x):
            return Ispline(x, k - 1, i + 1, knots_augmented)

        Ispline_vectorized = np.vectorize(Ispline_wrapper)
        column_list.append(Ispline_vectorized(xs).reshape(-1, 1))
    if verbose:
        print('\n', end='')  # Print \n after using \r in loop
    Phi = np.concatenate(column_list, axis=1)
    return Phi


def fit_model_decreasing(Phi, y):
    # Define the objective function
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


def fit_model_increasing(Phi, y):
    # Define the objective function
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


def main():
    """Main function for testing
    """
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
    num_points = 10_000
    xs = x_max * rng.random(num_points)

    # Test decreasing fit
    true_function = build_decreasing_function()
    probabilities = np.array(list(map(true_function, xs)))
    iterates = rng.random(num_points)
    ys = (iterates <= probabilities).astype(np.float64)

    num_knots = 5
    integrated_splines = IntegratedSplines(xs, ys, 'decreasing', num_knots, degree=4)
    integrated_splines.show()
    integrated_splines.show_design_matrix()

    # Test increasing fit
    true_function = build_increasing_function()
    probabilities = np.array(list(map(true_function, xs)))
    iterates = rng.random(num_points)
    ys = (iterates <= probabilities).astype(np.float64)

    num_knots = 5
    integrated_splines = IntegratedSplines(xs, ys, 'increasing', num_knots, degree=4)
    integrated_splines.show()
    integrated_splines.show_design_matrix()


if __name__ == '__main__':
    main()
