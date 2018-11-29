import numpy as np
from _decimal import Decimal
from numpy import linalg
'''
SOR iterative method.

Passes in Matrix A, b, uses zero vector for x as starting point. 

Requires an error to be passed in, in our case 10^-8.

Requires an actual solution to be be passed in to calculate norm.

Norm is calculated using built in linalg function.

The default w is 1.0 if the user does not input a factor.

SOR method in terms of matrix multiplication:
((D-wL)^-1) * Ux + ((D-wL)^-1) * b

This function modifies the above to:
w * L^-1 * (b - Ux), 
where L is a lower triangular matrix containing the diagonal,
and w is the SOR factor.
'''


def sor(A, b, x, error, actual_soln, max_iterations=9999, w=1.0):
    if error and actual_soln.any():
        # 'np.tril' returns a lower triangular matrix, containing the diagonal
        L = np.tril(A)
        # Multiply by a SOR factor
        L *= w
        U = A - L
        n = 0
        real_norm = linalg.norm(actual_soln)
        while True:
            x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
            norm = linalg.norm(x)
            n += 1
            if abs(Decimal(real_norm) - Decimal(norm)) <= error or n >= max_iterations:
                break
    else:
        raise SyntaxError("Must pass in an error term and actual solution.")

    # Returns an array of the solution vector and number of iterations
    return [x, n]
