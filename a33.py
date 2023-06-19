from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
import utils

# Problem name and setup
problem_name = 'a33'
utils.create_output_dir(problem_name)

## Reformulate the following problem as a convex optimization problem
## and solve it using CVXPY:

#%% Problem A3.3 a)
# norm([x + 2*y, x - y]) == 0

x = Variable()
y = Variable()
constraints = [x + y == 1, x - y >= 1]
obj = Minimize(square(x - y))
prob = Problem(obj, constraints)
print(prob)
