# monotone-splines
Model monotone conditional probabilities using M-splines, I-splines, and C-splines using CVXOPT.

## Introduction

At its core, statistical learning is about generalizing from a known data set to an unknown data set. This is difficult because it is inductive rather than deductive, and philosophers have long known about the problems and difficulties of induction. Mathematically speaking, we say the problem of recovering a distribution from a finite sample is *ill-posed* (See Vapnik, The Nature of Statistical Learning) and hence cannot be solved for exactly. However, like other ill-posed problems, we can introduce *regularization* to make the problem tractable. Often, this regularization takes the form of smoothness requirements (belonging to C^k for a fixed k), either strict requirements in the sense of taking the function class V to only contain functions of the required smoothness, or soft requirements in the form of a preference for smoother functions by incorporating a penalty on less-smooth functions in the loss function. This project shows how we can use monotone splines (M-splines, I-splines, and C-splines) to build a function class that strictly requires not only smoothness, but also can strictly require the functions to be either non-negative or non-positive (M-splines), non-decreasing or non-increasing (I-splines), and convex or concave (C-splines). Monotone splines were first introduced by J. O. Ramsay in (Ramsay, Monotone Regression Splines in Action).

## Modeling Monotone Conditional Probabilities

The specific problem solved here is to recover the function f(x) = P(y | x) where y is a binary response to a real-valued predictor x, where we know that f is non-decreasing. The restriction that f be non-decreasing turns out to be quite strong at improving the ability to generalize. The application in mind is to predict the case fatality rate (CFR) of covid-19 conditioned on age using the publically available data from https://github.com/jihoo-kim/Data-Science-for-COVID-19. DISCLAIMER: This is a toy example on a very limited amount of data collected early in the pandemic. Actual values of the CFR for South Korea are computed on age intervals and [published by the KCDC](https://www.cdc.go.kr/board/board.es?mid=a30402000000&bid=0030&act=view&list_no=366537). Note that this early data overestimates the CFR.

## Splines

Splines are an effective and simple tool for modeling smooth functions. Splines build upon polynomials, the prototypical example of smooth functions, but avoid the problem of global instability under local perturbations. A system of splines starts by partitioning an interval and then building a separate spline on each interval as a polynomial that smoothly connects to its neighbors. Monotone splines let us further require each of these polynomials to be monotone in each of the three senses listed above (corresponding to either M-splines, I-splines, or C-splines).

## Estimating the Model

To fit the model to the data, we use the ell_2 loss (squared error). However, the non-decreasing requirement also requires that the coefficients on each of the basis functions be non-negative. Fortunately, this problem can be cast as a quadratic programming problem (QP) which is solved using the CVXOPT function solvers.qp. 

## Computing Error Bars

To compute error bars on the estimated function, we can reestimate the model using bootstrapped data. By generating new, synthetic data sets by bootstrapping and then fitting the I-spline model, the error bars also conform to the restriction of non-decreasingness. Note how this gives us a high degree of confidence about the CFR for the very young, even though there are a small number of data points for ages close to 0.
