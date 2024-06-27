import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-2, 2, 500)

# Define coefficient parameters
c1 = 1.0

# Define modified objective functions with coefficient parameters
def f1_coefficient(x, c1):
    return c1 + x

def f2_coefficient(x, c1):
    return  (1 - np.sqrt(c1 + x))

# Calculate the coefficient-modified f1 and f2 values
f1_coefficient_values = f1_coefficient(x_values, c1)
f2_coefficient_values = f2_coefficient(x_values, c1)

# Plot the modified objective functions with coefficient parameters
plt.figure(figsize=(10, 6))

plt.plot(x_values, f1_coefficient_values, label=f'f1(x) = {c1} * x')
plt.plot(x_values, f2_coefficient_values, label=f'f2(x) =  1 - sqrt({c1} * x)')

plt.xlabel('x')
plt.ylabel('Objective Function Value')
plt.title('Coefficient-Modified ZDT1 Objective Functions')
plt.legend()
# plt.xlim(0, 1)
# plt.ylim(0, 1)
plt.grid(True)
plt.show()
