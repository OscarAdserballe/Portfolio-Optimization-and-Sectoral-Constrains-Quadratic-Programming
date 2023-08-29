# Portfolio-Optimization-and-Sectoral-Constrains-Quadratic-Programming
This Portfolio Optimization Tool attempts to look at comparative efficient frontiers given different constraints, to mimic some kind of risk analysis limiting exposure to, say, certain sectors as in this coded example. For the most part, it was just as an attempt to concretely apply some of the linear algebra and optimization from my Mathematical Methods for Quantitative Finance Course, and expanding on the given fictitious example provided there where it was by construciton a simplified and full rank matrix of assets, thereby side-stepping many of the problems you actually run into with real historical data.... as I unfortunately quickly discovered. 


## Background

### What is Portfolio Optimization?

Portfolio optimization is the science of selecting the best allocation of various assets to form a portfolio that achieves a particular investment objective. It aims to maximize expected returns while managing risk. Investors typically want to maximize their utility, which often means achieving high returns without exposing themselves to unnecessary risk. The concept was introduced by Harry Markowitz in 1952, for which he later won a Nobel Prize in Economics. In this tool, we implement Modern Portfolio Theory (MPT) as a foundation for the optimization.

### What is Efficient Frontier?

The efficient frontier is a crucial concept in portfolio optimization. It represents a curve on a graph where the x-axis represents portfolio risk and the y-axis represents expected return. Any point on this curve signifies an 'optimal' portfolio, meaning that it offers the highest possible expected return for a given level of risk. Positions below the curve are considered suboptimal because they provide less return for a given level of risk. Points above the curve are unattainable with the given asset set.

### Mathematics

The portfolio optimization process uses several mathematical models to arrive at the optimal portfolio. The primary objective function is to minimize the portfolio's variance, mathematically defined as:

\[
\sigma_p^2 = w^T C w
\]

Where:
- \( \sigma_p^2 \) is the portfolio variance.
- \( w \) is the weight vector for the assets in the portfolio.
- \( C \) is the covariance matrix of the asset returns.

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