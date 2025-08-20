# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# --- 1. Data Preparation ---

# Define the true weight and bias for our linear relationship
# These are the parameters our model will try to learn
weight = torch.tensor(0.7)
bias = torch.tensor(0.3)

# Define the start, end, and step size for our data points
start = 0
end = 1
step = 0.02

# Create the feature tensor X (input data)
# .unsqueeze(1) adds an extra dimension to make it a column vector (required for nn.Linear)
X = torch.arange(start, end, step).unsqueeze(1)

# Create the target tensor y (labels) using the linear formula y = weight * X + bias
y = X * weight + bias

# --- 2. Train/Test Split ---

# Calculate the index for an 80/20 split
train_test_split = int(0.8 * len(X))

# Split the data into training and testing sets
x_train, y_train = X[:train_test_split], y[:train_test_split]
x_test, y_test = X[train_test_split:], y[train_test_split:]

# Print the lengths of the test and train sets to verify the split
print(f"Length of test set: {len(x_test)}, Length of train set: {len(x_train)}")

# --- 3. Device Configuration ---

# Check if a CUDA-enabled GPU is available, otherwise use the CPU
# This allows the code to run on a GPU for faster computation if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 4. Visualization Function ---

# Define a function to plot the data and predictions
def plot_prediction(x_train=x_train, 
                    y_train=y_train, 
                    x_test=x_test, 
                    y_test=y_test, 
                    prediction=None):
    """
    Plots training data, test data, and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    
    # Plot training data in blue
    plt.scatter(x_train, y_train, c='b', s=10, label="Training data")
    
    # Plot testing data in green
    plt.scatter(x_test, y_test, c='g', s=10, label="Testing data")
    
    # If predictions are provided, plot them in red
    if prediction is not None:
        # .cpu() is used to move the tensor to the CPU for plotting if it was on the GPU
        plt.scatter(x_test, prediction.cpu(), c='r', s=10, label="Prediction")
        
    plt.legend(prop={'size': 14})
    plt.title("Training vs Testing vs Prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Example of how to call the plotting function (currently commented out)
# plot_prediction()

# --- 5. Model Definition ---

# Create a Linear Regression model class that inherits from nn.Module
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the built-in nn.Linear layer for simplicity and efficiency
        # in_features=1: The model expects one input feature (our X value)
        # out_features=1: The model produces one output feature (our predicted y value)
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    
    # Define the forward pass of the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input tensor 'x' through the linear layer
        return self.linear_layer(x)

# --- 6. Model Initialization, Loss, and Optimizer ---

# Set a manual seed for reproducibility
torch.manual_seed(42)

# Create an instance of the model
model = LinearRegression()

# Define the loss function: Mean Absolute Error (L1 Loss)
loss_fn = nn.L1Loss()

# Define the optimizer: Stochastic Gradient Descent (SGD)
# It will update the model's parameters (weights and bias) to minimize the loss
# lr=0.02 is the learning rate, which controls the step size of the updates
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.02)

# --- 7. Training and Evaluation Loop ---

# Set the number of training epochs (how many times to loop over the entire training dataset)
epochs = 200

for epoch in range(epochs):
    
    ### Training ###
    # Set the model to training mode
    model.train()
    
    # 1. Forward pass: Make predictions on the training data
    y_pred = model(x_train)
    
    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)
    
    # 3. Zero the gradients of the optimizer to prevent them from accumulating
    optimizer.zero_grad()
    
    # 4. Backward pass: Compute the gradients of the loss with respect to model parameters
    loss.backward()
    
    # 5. Update the parameters: Adjust the model's weights and bias
    optimizer.step()
    
    ### Evaluation ###
    # Set the model to evaluation mode
    model.eval()
    
    # Use torch.inference_mode() as a context manager to disable gradient calculation,
    # which makes evaluation faster and more memory-efficient.
    with torch.inference_mode():
        # 1. Forward pass on the test data
        test_pred = model(x_test)
        
        # 2. Calculate the loss on the test data
        test_loss = loss_fn(test_pred, y_test)
    
    # Print the training loss every 20 epochs to monitor progress
    if epoch % 20 == 0:
        print(f"Epoch: {epoch} | Training Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

# --- 8. Post-Training ---

# Print the final learned parameters (state dictionary) of the model
print("\nFinal model state dictionary:")
print(model.state_dict())

# Plot the final predictions against the actual test data
plot_prediction(prediction=test_pred)


# --- 9. Saving and Loading the Model ---

# Define a function to save the model's state dictionary
def saver(model, path):
    """Saves the model's state_dict to a file."""
    torch.save(obj=model.state_dict(), f=path)

# Save the trained model to a file named "Nadeem.pt"
model_path = "Nadeem.pt"
saver(model, model_path)
print(f"\nGood job, your model has been saved to {model_path} <3")

# --- 10. File System Check (Optional) ---
# This section uses a command-line tool to check if the file was created.

import subprocess
try:
    # On Windows, 'dir' lists directory contents. On Linux/macOS, use 'ls'.
    # This command tries to list the contents of a 'module' directory, which might not be what's intended.
    # A better command might be 'dir' or 'ls' on its own to see the saved file in the current directory.
    result = subprocess.run(['dir'], capture_output=True, text=True, shell=True, check=True)
    print('\nCurrent directory contents:')
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"An error occurred while checking the directory.")
    print(e.stdout)

# --- 11. Loading the Saved Model ---

# Create a new, untrained instance of the LinearRegression model
load_model = LinearRegression()

# Load the saved parameters (state dictionary) into the new model instance
load_model.load_state_dict(torch.load(f=model_path))

# Print the state dictionary of the loaded model to confirm it matches the saved one
print(f"\nLoaded model state dictionary:")
print(load_model.state_dict())

# Display any plots that have been created
plt.show()
