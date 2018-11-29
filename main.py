import math
import numpy as np
from Jacobi import jacobi
from Printer import print_results
from Gauss_Seidel import gauss_seidel
from SOR import sor
'''
MATH 5336, Homework #6

Main function area, run program from here.

A and b matrices were created from HW #6 description.

We will use x = [0,0,0,0,0,0,0,0] as starting value

Prints number of iterations that each method took along with their X values

Printing will be done with the help of a helper printer file
'''


A = np.array([[-1, 0, 0, math.sqrt(2)/2, 1, 0, 0, 0],
    [0, -1, 0, math.sqrt(2)/2, 0, 0, 0, 0],
   [0, 0, -1, 0, 0, 0, 1/2, 0],
   [0, 0, 0, -math.sqrt(2)/2, 0, -1, -1/2, 0],
   [0, 0, 0, 0, -1, 0, 0, 1],
   [0, 0, 0, 0, 0, 1, 0, 0],
   [0, 0, 0, -math.sqrt(2)/2, 0, 0, math.sqrt(3)/2, 0],
   [0, 0, 0, 0, 0, 0, math.sqrt(3)/2, -1]
   ])
b = np.array([0,0,0,0,0,10000,0,0])
x = np.array([0,0,0,0,0,0,0,0])

# Uses built-in linalg solver to find X
x_solved = np.dot(np.linalg.inv(A), b)

# Print actual solution for comparison
print("The actual solution using the built-in matrix solver is: ")
print_results(x_solved)

# Runs Jacobi's method and prints its results
x_jacobi = jacobi(A, b, x, error=10**-8, actual_soln = x_solved)
print()
print("Jacobi's method took {} iterations: " .format(x_jacobi[1]))
print_results(x_jacobi[0])

# Runs Gauss Seidel method and prints its results
x_gauss_seidel = gauss_seidel(A, b, x, error=10**-8, actual_soln=x_solved)
print()
print("Gauss Seidel method took {} iterations: ".format(x_gauss_seidel[1]))
print_results(x_gauss_seidel[0])

# Runs SOR with w = 1.25 and prints its results
x_sor = sor(A, b, x, error=10**-8, actual_soln=x_solved, w=1.25)
print()
print("SOR method took {} iterations: ".format(x_sor[1]))
print_results(x_sor[0])
