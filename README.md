PyTorch Linear Regression Quickstart 🚀
Welcome! This project provides a clean and straightforward implementation of a linear regression model using PyTorch. It's designed as a fundamental example for beginners to grasp the core concepts of building, training, and evaluating a neural network for a simple regression task.
📋 Table of Contents
* Description
* Key Features
* Requirements
* How to Run
* Code Breakdown
* Expected Output
* Contributing
📝 Description
The script systematically walks through a complete machine learning workflow:
1. Data Generation: Creates a synthetic dataset based on a clear linear relationship (y = weight * x + bias).
2. Data Splitting: Partitions the data into training and testing sets for robust evaluation.
3. Model Definition: Implements a simple linear regression model using PyTorch's nn.Module.
4. Model Training: Trains the model using Stochastic Gradient Descent (SGD) to minimize the L1 Loss (Mean Absolute Error).
5. Visualization: Renders the results with Matplotlib, showing the training data, testing data, and the model's predictions.
6. Model Persistence: Saves the trained model's state and demonstrates how to load it back for inference.
✨ Key Features
* Pure PyTorch Implementation: Built entirely using the PyTorch library.
* Train/Test Split: Separates data for proper model evaluation.
* Custom Model Class: Defines a neural network using the standard nn.Module class.
* Clear Visualization: Includes a helper function to plot the results for easy interpretation.
* Model Saving & Loading: Demonstrates how to save and load model weights effectively.
⚙️ Requirements
To run this project, you need Python installed along with the following libraries:
* torch
* numpy
* matplotlib
You can install them easily using pip:
pip install torch numpy matplotlib

🚀 How to Run
1. Clone the repository or save the Python script to your local machine.
2. Open your terminal or command prompt.
3. Navigate to the project directory:
cd path/to/your/project

4. Run the script:
python your_script_name.py

(Replace your_script_name.py with the actual name of your file.)
💻 Code Breakdown
   * Data Preparation: A simple dataset is created using torch.arange and a known weight and bias.
   * plot_prediction() function: A utility function to visualize the dataset and model predictions.
   * LinearRegression class: The core model, which inherits from nn.Module and contains a single nn.Linear layer.
   * Training Loop: The model is trained for a set number of epochs. In each epoch, it performs a forward pass, calculates the loss, performs backpropagation, and updates the weights.
   * Evaluation: The model's performance is periodically checked on the test set.
   * Saving/Loading: The model's learned parameters (state_dict) are saved to a file and then loaded into a new model instance to verify the process.
📊 Expected Output
When you run the script, you will see:
   1. Console Output: The script will print the training and test loss, showing the model's improvement over time.
Epoch: 0 | Training Loss: 0.28... | Test Loss: 0.42...
Epoch: 20 | Training Loss: 0.11... | Test Loss: 0.22...
...
Epoch: 180 | Training Loss: 0.00... | Test Loss: 0.00...

   2. Final Model Parameters: The learned weights and bias of the model will be printed.
   3. Plot Window: A Matplotlib window will open, displaying a scatter plot of the training data (blue), testing data (green), and the model's final predictions (red).
🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.