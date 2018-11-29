x_variables = [
    "F1", "F2", "F3", "f1", "f2", "f3", "f4", "f5"
]


# Helper function to print the X values in a more readable form
def print_results(matrix, variables=x_variables):
    for i in range(len(matrix)):
        print("  {} = {}".format(variables[i], matrix[i]))
