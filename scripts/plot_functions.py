import numpy as np
import matplotlib.pyplot as plt

# Define the function f  (for 1-d optimization)
# Schaffer
def f(x, shift):
    f1 = (shift*x)**2
    f2 = (shift*x - 2)**2
    return f1, f2
# Define a range for x
# Schaffer
x_values = np.linspace(0, 2, 100)
shifts = [0.9, 1.2, 1.5, 3]


# #ZDT1
# def f(x, shift):
#     x = np.array(x)
#     f1 = x * shift
#     f2 = 1 - np.sqrt(x * shift)
#     return np.array([f1, f2])

# # Define a range for x - ZDT1
# x_values = np.linspace(0, 1, 100)
# shifts = [0.9, 1.2, 1.5, 3]

# #ZDT2
# def f(x, shift): 
#     x = np.array(x)
#     f1 = x * shift
#     f2 = 1 - (x * shift) ** 2
#     return np.array([f1, f2])

# # Define a range for x - ZDT2
# x_values = np.linspace(0, 1, 100)
# shifts = [0.9, 1.2, 1.5, 3]

# # FON
# def f(x, shift): 
#     n = len(x)
#     x = np.array(x)
#     f1 = 1 - np.exp(-np.sum((x + shift - 1 / np.sqrt(n)) ** 2))
#     f2 = 1 - np.exp(-np.sum((x + shift + 1 / np.sqrt(n)) ** 2))
    
#     return np.array([f1, f2])

# # Define a range for x - FON
# x_values = np.linspace(-4, 4, 100)
# shifts = [0.9, 1.2, 1.5, 3]


# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.flatten()

# Plot for each shift value
for i, shift in enumerate(shifts):
    f1_values = []
    f2_values = []
    for x in x_values:
        x_vec = np.array([x])  # Create a vector of length 1
        f1, f2 = f(x_vec, shift)
        f1_values.append(f1)
        f2_values.append(f2)

    axs[i].plot(x_values, f1_values, label='f1')
    axs[i].plot(x_values, f2_values, label='f2')
    axs[i].set_title(f'Shift = {shift}')
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('f1, f2')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
