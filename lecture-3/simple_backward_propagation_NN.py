# Importing the required library
from copy import error
import numpy as np
import main_lib.plotting as plotting
import main_lib.analysis_utilities as analysis_utilities


def initialize_network_parameters(lr = None, epochs = None):
    print("Initialized weights and biases:")
    num_inputs = 2 # number of inputs
    num_hidden_layers = 2 # number of hidden layers
    num_nodes_output = 1 # number of nodes in the output layer
    num_nodes_hidden = [2, 2] # number of nodes in each hidden layer
    # Visualize the network architecture
    plotting.draw_neural_net(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)

    # Initialize weights and biases randomly within the range [-1, 1]
    w1 = np.random.rand(num_hidden_layers, num_inputs) * 2 - 1  # Weights from input to hidden layer
    b1 = np.random.rand(num_hidden_layers, 1) * 2 - 1          # Bias for hidden layer
    w2 = np.random.rand(num_nodes_output, num_hidden_layers) * 2 - 1 # Weights from hidden to output layer
    b2 = np.random.rand(num_nodes_output, 1) * 2 - 1          # Bias for output layer
    print("Weights from input to hidden layer (w1):\n", w1)
    print("Bias for hidden layer (b1):\n", b1)
    print("Weights from hidden to output layer (w2):\n", w2)
    print("Bias for output layer (b2):\n", b2)
    return w1, b1, w2, b2, lr, epochs


def train_network(X, d, w1, b1, w2, b2, lr, epochs):
    error_list = []
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
        a1 = analysis_utilities.sigmoid(z1)  # Sigmoid activation for hidden layer

        z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
        a2 = analysis_utilities.sigmoid(z2)  # Sigmoid activation for output layer

        # Error calculation and backpropagation
        # de/dw2 = de/da2 .da2/dz2 . dz2/dw2
        error = d - a2  # Difference between expected and actual output
        # de/da2 .da2/dz2 = (d - a2) * (a2 * (1 - a2))
        da2 = error * (a2 * (1 - a2))  # Derivative for output layer de/da2 .da2/dz2
        dz2 = da2  # Gradient (error) for output layer

        # Propagate error to hidden layer
        # de/da1 = de/dz2 . dz2/da1
        da1 = np.dot(w2.T, dz2)  # Gradient for hidden layer da1/dz1
        dz1 = da1 * (a1 * (1 - a1))  # Derivative for hidden layer da1/dz1

        # Update weights and biases
        #w2 -> w2 - (eta * de/dw2) = w2 -> w2 - (eta * de/da2 .da2/dz2 . dz2/dw2) 

        #b2 -> b2 - (eta * de/db2)
        # This  summation is computed via matrix multiplication:
        w2 += lr * np.dot(dz2, a1.T)  # Update weights from hidden to output layer (eta * de/dw2)
        b2 += lr * np.sum(dz2, axis=1, keepdims=True)  # Update bias for output layer (eta * de/db2)

        w1 += lr * np.dot(dz1, X.T)  # Update weights from input to hidden layer (eta * de/dw1)
        b1 += lr * np.sum(dz1, axis=1, keepdims=True)  # Update bias for hidden layer (eta * de/db1)
        if (epoch+1)%10000 == 0:
                print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
                error_list.append(np.average(abs(error)))
    plotting.plot_error_over_epochs(error_list, output_path="output/")
    return w1, b1, w2, b2

# Defining inputs and expected output (XOR truth table)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2x4 matrix, each column is a training example
d = np.array([0, 1, 1, 0])  # Expected output for XOR

plotting.plot_activation_functions(z = np.linspace(-10, 10, 400),
                               output_path="output/")

# Initialize network and parameters
w1, b1, w2, b2, lr, epochs = initialize_network_parameters(lr = 0.1, epochs = 1800000)

# Training the network using backpropagation
w1, b1, w2, b2 =train_network(X, d, w1, b1, w2, b2, lr, epochs)

# Testing the trained network
z1 = np.dot(w1, X) + b1  # Weighted sum for hidden layer
a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation for hidden layer

z2 = np.dot(w2, a1) + b2  # Weighted sum for output layer
a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation for output layer
error = d - a2  # Difference between expected and actual output

# Print results
print('Final output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))


