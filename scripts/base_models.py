import numpy as np
import pandas as pd
import yaml

args = yaml.safe_load(open('configs/test_function.yml','r'))

def get_std_dev(shift):
    
    samples = np.linspace(start=args['Optimization']['range'][0], 
                                   stop=args['Optimization']['range'][1],
                                   num=100)
    
    output1 = []
    output2 = []


    for sample in samples:
        result = f(sample, shift)
        output1.append(result[0])
        output2.append(result[1])

    output1 = np.array(output1)
    output2 = np.array(output2)
    print(output1.shape, output2.shape)
    
    return np.std(output1), np.std(output2)

# Define the ZDT1 function with shift parameter
# def ZDT1(x, shift):
#     x = np.array(x)
#     g = 1 + ((9 - shift) / (len(x) - 1)) * np.sum(x[1:])
#     f1 = x[0] * (1 - shift)
#     f2 = g * (1 - np.sqrt(x[0] / g))
#     return np.array([f1, f2])

# def f(x, shift=0): #ZDT1 1-D
#     x = np.array(x)

#     # shift  = 1.0

#     f1 = x * shift
#     f2 = 1 - np.sqrt(x * shift)

#     return np.array([f1, f2])

# def f(x, shift):
#     return x * np.sin(x + np.pi + shift) + 0.1 * x

def f(x, shift): #fronesca and fleming
    n = len(x)
    x = np.array(x)
    # shift = 0.0
    f1 = 1 - np.exp(-np.sum((x + shift - 1 / np.sqrt(n)) ** 2))
    f2 = 1 - np.exp(-np.sum((x + shift + 1 / np.sqrt(n)) ** 2))
    return np.array([f1, f2])

# Define shift values and number of samples
shift_values = [-1.0, 0.5, 1.5, 2]
num_samples = 20

# Randomly sample points from the function
data = []

# Create separate CSV files for each shift value
file_paths = []



for i, shift in enumerate(shift_values):
    data = []
    for _ in range(num_samples):
        noise1, noise2 = get_std_dev(shift)
        x = np.random.uniform(args['Optimization']['range'][0], args['Optimization']['range'][1], 1)  
        #x = np.random.rand(2) # 2D array with values between 0 and 1
        f_values = f(x, shift)
        # data.append([x[0], x[1], f_values[0], f_values[1]])
        data.append([x[0], f_values[0] + np.random.normal(0, noise1), f_values[1] + np.random.normal(0, noise2)])
        # data.append([x[0], f_values[0], f_values[1]])
        # data.append([x[0], f_values[0]])
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to individual CSV files without headers and with whitespace separator
    file_name = f"data.csv"
    file_path = f"base_models\model{i+1}\{file_name}"
    df.to_csv(file_path, header=False, index=False, sep=' ')
    file_paths.append(file_path)

file_paths
