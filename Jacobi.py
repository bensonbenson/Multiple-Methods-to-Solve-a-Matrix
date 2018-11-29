import numpy as np
from _decimal import Decimal
from numpy import linalg
'''
Jacobi's iterative method.

Passes in Matrix A, b, uses zero vector for x as starting point. 

Requires an error to be passed in, in our case 10^-8.

Requires an actual solution to be be passed in to calculate norm.

Norm is calculated using built in linalg function.

Jacobi's method in terms of matrix multiplication:
D^-1 * (L-U) * x + D^-1 * b

This function modifies the above to:
D^-1 * (b-Rx),
where R is the remainder matrix, containing lower and upper triangular matrices
'''


def jacobi(A, b, x, error, actual_soln, max_iterations=9999):
    if error and actual_soln.any():
        D = np.diag(A)
        # R is the remainder matrix, containing lower and upper triangular matrices
        R = A - np.diagflat(D)
        n = 0
        real_norm = linalg.norm(actual_soln)
        while True:
            x = (b - np.dot(R, x)) / D
            norm = linalg.norm(x)
            n += 1
            if abs(Decimal(real_norm) - Decimal(norm)) <= error or n >= max_iterations:
                break
    else:
        raise SyntaxError("Must pass in an error term and actual solution.")

    # Returns an array of the solution vector and number of iterations
    return [x, n]
