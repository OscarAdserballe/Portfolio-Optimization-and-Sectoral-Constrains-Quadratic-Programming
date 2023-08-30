# Portfolio-Optimization-and-Sectoral-Constrains-Quadratic-Programming
This Portfolio Optimization Tool attempts to look at comparative efficient frontiers given different constraints, to mimic some kind of risk analysis limiting exposure to, say, certain sectors as in this coded example. For the most part, it was just as an attempt to concretely apply some of the linear algebra and optimization from my Mathematical Methods for Quantitative Finance Course, and expanding on the given fictitious example provided there where it was by construciton a simplified and full rank matrix of assets, thereby side-stepping many of the problems you actually run into with real historical data.... as I unfortunately quickly discovered. 


## Background

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