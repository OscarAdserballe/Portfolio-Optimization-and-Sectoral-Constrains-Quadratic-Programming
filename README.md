# Portfolio-Optimization-and-Sectoral-Constrains-Quadratic-Programming
This Portfolio Optimization Tool attempts to look at comparative efficient frontiers given different constraints, to mimic some kind of risk analysis limiting exposure to, say, certain sectors as in this coded example. For the most part, it was just as an attempt to concretely apply some of the linear algebra and optimization from my Mathematical Methods for Quantitative Finance Course, and expanding on the given fictitious example provided there where it was by construciton a simplified and full rank matrix of assets, thereby side-stepping many of the problems you actually run into with real historical data.... as I unfortunately quickly discovered. 


## Background

Workhorse model is as always just Lagrangian Optimization, simply, switching a bit from the notation employed in 18.02, 

$$
L(x, y, \lambda) = f(x, y) - \lambda(g(x, y)-c), for \space g= c
$$

Then just taking the gradient of it.

************Minimum Variance Portfolio Problem************

Defining the key moments for mean-variance optimization,

$$
\mu_p = \mu^Tw
$$

$$
\sigma_p^2 = w^TCw \space (as \space Var(\sum w R_i) = \sum w^2 C)
$$

And the single constraint we have for a moment, is that the weight of our portfolio exactly has to equal one, i.e. also indicating we can use our Lagrangian method.

$$
Const. \sum w_i = 1
$$

What is especially important to note is the fact that the Covariance matrix is positive-definite as long as we exclude risk-free and linearly dependent assets. (Side-note: Jesus Christ this step is hard)

Using Lagrangian optimization, our minimum-Variance portfolio can be found by,

$$
L(w, l) = 0.5w^TCw + l(1-\iota^Tw)
$$

(The latter expression is just another means of expressing that same constraint as before)

$$
\frac {\partial L}{ \partial w_i} = \sum_j w_jC_{ij} - l\iota = 0
$$

Taking the partial derivative is always damn hard with matrices, but what we’re saying here is that only the i’th row of the Covariance matrix survives.

Sol. that works for all partial derivatives with respect to individual weights,

$$
w= lC^{-1}\iota
$$

Eliminating the lagrangian multiplier using the constraint

$$
\iota^T w = l(\iota^Tc^{-1}\iota)=1
$$

Both terms, the inverted covariance matrix surrounded by the iotas and the lagrangian multiplier are both just scalars, Whereby we can get the lagrangian, and from there get the weights vector.

$$
l = \frac{1}{\iota^Tc^{-1}\iota} = \sigma^2_{min}
$$

$$
w_{min} = \frac{1}{\iota^Tc^{-1}\iota}C^{-1}\iota
$$

****************Adding A Return Constraint: Getting the Efficient Market Frontier****************

The step to solve for the Markowitz Efficient frontier is just one more step, and a bit more mathematics. We solve for the minimum variance for a given point of return for every single potential return to solve for the efficient frontier. New Lagrangian, 

$$
L(w, l, m) = \frac{1}{2}w^TCw+l(1-C^Tw) + m(\mu_p-\mu^Tw)
$$

Yada, yada, we do the same kind of Lagrangian Optimization as before,

$$
w = C^{-1}(l\iota + m\mu)
$$

We set the constraints and reduce it all to another matrix equation, where M just denotes a matrix of scalar values as given by the equations.

$$
\iota^Tw = 1 = l(\iota ^TC^{-1}\iota)+m(\mu^TC^{-1}\iota)
$$

$$
\mu_P^Tw = \mu_P = l(\mu ^TC^{-1}\iota)+m(\mu^TC^{-1}\mu)
$$

$$
\begin{pmatrix} 1 \\\ \mu_p \end{pmatrix} = M\begin{pmatrix} l \\\ m \end{pmatrix}
$$

$$
\begin{pmatrix} l \\ m \end{pmatrix} = M^{-1}\begin{pmatrix} 1 \\ \mu_p \end{pmatrix}
$$

And from here we can get an expression for the minimum-variance portfolio for some given $\mu_p$

$$
\sigma^2_p=w^TCw = (l \space \space m) M\begin{pmatrix} l \\\ m \end{pmatrix}
$$

Thus the analytic solution to our Markowitz Efficient Frontier, and the beauty lies exactly in the ability to reduce these incredibly high-dimensional portfolios down to a 2D-plot focusing on two moments. While it’s no doubt an egregious oversimplification, the parsimony of the plot is beautiful in itself in just how much it communciates.

Just going back to our expression for the portfolio weights, we notice that it’s exactly a linear combination of our minimum variance portfolio and the maximum Sharpe Ratio portfolio

$$
w = C^{-1}(l\iota + m\mu)
$$

**********Quadratic Programming - Optimization for a Broader Class of Constraints**********

But the issue lies exactly in the form it requires us to render the constraints, namely a strict equal sign. Becomes incapable once we need to optimise for inequality constraints, that act as kind of soft constraints.

The actual mathematics of it, is very simple. Using a 2nd order Taylor Series, we can approximate the value of an optimum,

$$
f(x) \approx f(x_0) + \frac{1}{2} f''(x)(x-x_0)  \rightarrow i.e. f'(x_0) = 0 \space (c.p.)
$$

In multivariate case, we set the gradient to zero instead, as well as a matrix, Q, the Hessian matrix, of the second-order derivatives

$$
f(x) \approx f(x_0) + \frac{1}{2}(x-x_0)^TQ(x-x_0)
$$

And it’s on examination of the eigenvalues of the Hessian we can see the type of optima:

- $\lambda s<= 0$ → Convex and a Minimum
- $\lambda s\le 0$  → concave and a Maximum
- $\lambda s= 0$ → Indicating a flat direction for the corresponding eigenvector
- Otherwise indicating a Saddle Point

**********************************Beyond Markowitz: What Can we Do with Quadratic Programming?**********************************

Using Quadratic porgramming, we can start adding additional constraints, like, imposing the mandate it’s a long-only portfolio, or that we reduce our exposure to some sector. An interesting comparison lies exactly in comparing the efficient frontier with and without constriants. Quadratic programming is therefore an excellent tool for risk analyses, bounding exposure of our portfolios.

### What is Portfolio Optimization?

Portfolio optimization is the science of selecting the best allocation of various assets to form a portfolio that achieves a particular investment objective. It aims to maximize expected returns while managing risk. Investors typically want to maximize their utility, which often means achieving high returns without exposing themselves to unnecessary risk. The concept was introduced by Harry Markowitz in 1952, for which he later won a Nobel Prize in Economics. In this tool, we implement Modern Portfolio Theory (MPT) as a foundation for the optimization.

### What is the Efficient Frontier?

The efficient frontier is a crucial concept in portfolio optimization. It represents a curve on a graph where the x-axis represents portfolio risk and the y-axis represents expected return. Any point on this curve signifies an 'optimal' portfolio, meaning that it offers the highest possible expected return for a given level of risk. Positions below the curve are considered suboptimal because they provide less return for a given level of risk. Points above the curve are unattainable with the given asset set.

### Mathematics


The portfolio optimization process uses several mathematical models to arrive at the optimal portfolio. The primary objective function is to minimize the portfolio's variance, mathematically defined as:

&sigma;<sub>p</sub><sup>2</sup> = w<sup>T</sup>Cw

Where:
- &sigma;<sub>p</sub><sup>2</sup> is the portfolio variance.
- w is the weight vector for the assets in the portfolio.
- C is the covariance matrix of the asset returns.

In code, this objective function could be implemented using libraries such as NumPy and SciPy:

```python
import numpy as np
from scipy.optimize import minimize

# Define the objective function (minimize portfolio variance)
def objective(w, C):
    return w.T @ C @ w

# Covariance matrix
C = np.array([[...]])

# Initial guess
initial_w = np.array([1./n]*n)

# Constraints
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Perform the optimization
result = minimize(fun=objective, x0=initial_w, args=(C,), constraints=constraints)
```

# Results

Using, then, historical S&P 500 data, we then try to analyse how the efficient frontier we're able to achieve shifts as we impose sectoral constraints.

First, though, substantial work is needed to tailor the data, so we can render the Covariance matrix actually invertible. This was a long process, regularization and removing highly correlated stocks from the dataset eventually did end up working. There must have been a bit way to do this, though, probably with something like PCA to capture the most important dimensions of the data, but I just couldn't quite get it to work, thus the suboptimal solution. Note for example, the incredibly highly correlated values as seen below.

<img src="\images\corr.png">

But with the transformations finally applied we can turn to the optimization problem, and we get the following efficient frontier with no constraints, where it's a simple question of Lagrangian optimizaiton. 

<img src="\images\eff.png">

And then we can start applying constraints to it, and compare the envelopes of the respective efficient frontiers. We should expect the efficient frontier to be smaller as we start imposing inequality constraints like restricting the exposure to the financial and tech sector, and that is exactly what we see. Just, surprisingly, the effect size doesn't seem to be very great. Exactly here when we start applying inequality constraints like stating portfolio exposure has to be less than or equal to some x, lagrangian optimisation simply doesn't work, so we use an implementation of some quadratic programming from the SciPy library.

<img src="\images\eff_constrained.png">
