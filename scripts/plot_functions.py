# import matplotlib.pyplot as plt
# import numpy as np

# # Define the function f for ZDT1 with shift parameter
# def f(x, shift):
#     x = np.array(x)
#     return np.array([x * shift, 1 - np.sqrt(x * shift)])

# # Generate values for x
# x_values = np.linspace(0, 1, 400)

# # Define different shift parameters
# shifts = [0.5, 1.0, 1.5, 2.0]

# # Create subplots
# fig, axs = plt.subplots(4, 2, figsize=(14, 16))

# # Plot for each shift parameter
# for i, shift in enumerate(shifts):
#     f1_values, f2_values = f(x_values, shift)
    
#     # Plot the function values
#     axs[i, 0].plot(x_values, f1_values, label='$f_1(x)$')
#     axs[i, 0].plot(x_values, f2_values, label='$f_2(x)$')
#     axs[i, 0].set_title(f'Function Values for Shift = {shift}')
#     axs[i, 0].set_xlabel('$x$')
#     axs[i, 0].set_ylabel('Function Values')
#     axs[i, 0].grid(True)
#     axs[i, 0].legend()
    
#     # Plot the Pareto front
#     axs[i, 1].plot(f1_values, f2_values, label=f'Shift = {shift}')
#     axs[i, 1].set_title(f'Pareto Front for Shift = {shift}')
#     axs[i, 1].set_xlabel('$f_1(x)$')
#     axs[i, 1].set_ylabel('$f_2(x)$')
#     axs[i, 1].grid(True)
#     axs[i, 1].legend()

# plt.tight_layout()
# plt.show()

# Define new shift parameters
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the function f for ZDT2 with shift parameter for 2D x
def f(x, shift):
    x = np.array(x)
    g = 1 + ((9 - shift) / (len(x) - 1)) * np.sum(x[1:])

    f1 = x[0] * (1 - shift)
    f2 = g * (1 - (x[0] / g) ** 2)

    return np.array([f1, f2])

# Generate values for x
x0_values = np.linspace(0, 1, 100)
x1_values = np.linspace(0, 1, 100)
x0, x1 = np.meshgrid(x0_values, x1_values)
x_flat = np.vstack((x0.flatten(), x1.flatten()))

# Define shift parameters
shifts = [0.0, -3.0, -6.0, -9.0]

# Create subplots for 3D plots
fig = plt.figure(figsize=(20, 16))

for i, shift in enumerate(shifts):
    f1_values = []
    f2_values = []
    
    for j in range(x_flat.shape[1]):
        f1, f2 = f(x_flat[:, j], shift)
        f1_values.append(f1)
        f2_values.append(f2)
    
    f1_values = np.array(f1_values).reshape(x0.shape)
    f2_values = np.array(f2_values).reshape(x0.shape)
    
    # Plot the function values
    ax = fig.add_subplot(4, 2, 2*i+1, projection='3d')
    ax.plot_surface(x0, x1, f1_values, cmap='viridis')
    ax.set_title(f'Function $f_1$ for Shift = {shift}')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$f_1(x)$')
    
    ax = fig.add_subplot(4, 2, 2*i+2, projection='3d')
    ax.plot_surface(x0, x1, f2_values, cmap='viridis')
    ax.set_title(f'Function $f_2$ for Shift = {shift}')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$f_2(x)$')

plt.tight_layout()
plt.show()
