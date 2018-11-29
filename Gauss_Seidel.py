import numpy as np
from _decimal import Decimal
from numpy import linalg
'''
Gauss Seidel iterative method.

Passes in Matrix A, b, uses zero vector for x as starting point. 

Requires an error to be passed in, in our case 10^-8.

Requires an actual solution to be be passed in to calculate norm.

Norm is calculated using built in linalg function.

Gauss Seidel method in terms of matrix multiplication:
((D-L)^-1) * Ux + ((D-L)^-1) * b

This function modifies the above to:
L^-1 * (b - Ux), 
where L is a lower triangular matrix containing the diagonal
'''


def gauss_seidel(A, b, x, error, actual_soln, max_iterations=9999):
    if error and actual_soln.any():
        # 'np.tril' returns a lower triangular matrix, containing the diagonal
        L = np.tril(A)
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
