import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Load the configuration file
args = yaml.safe_load(open('configs/test_function.yml', 'r'))

weights_arr = []

for i in range(3, args['Optimization']['n_steps'] + 1):
    filename = f'models/iter_{i}/multi_model.pth'
    
    # Load the saved data
    save_data = torch.load(filename)
    
    # Access the weights array
    weights = save_data['weights']
    
    # Convert the weights to numpy arrays
    weights_numpy = [w.detach().cpu().numpy() for w in weights]
    
    # Store the weights for this iteration
    weights_arr.append(weights_numpy)
    
    print(f"Iteration {i} weights:", weights_numpy)

# Convert the list of weights to a numpy array for easier manipulation
weights_arr = np.array(weights_arr)

# Plotting the weights vs iterations for all models
iterations = np.arange(3, args['Optimization']['n_steps'] + 1)
num_models = len(weights_arr[0])  # Number of models
num_weights = weights_arr.shape[2]
plt.figure(figsize=(15, 5 * num_models))

for model_index in range(num_models):
    plt.subplot(num_models, 1, model_index + 1)
    
    for weight_index in range(num_weights):
        # Extract the weights for the current model and current weight
        weights_model_weight = weights_arr[:, model_index, weight_index]
        
        if weight_index < num_weights - 1:
            plt.plot(iterations, weights_model_weight, marker='o', linestyle='-', label=f'base model {weight_index + 1}')
        else:
            plt.plot(iterations, weights_model_weight, marker='o', linestyle='-', label=f'target model')
    
    plt.xlabel('Iteration')
    plt.ylabel('Weight Value')
    plt.title(f'Weights vs Iterations for objective {model_index + 1}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
