import matplotlib.pyplot as plt
import numpy as np
import main_lib.analysis_utilities as analysis_utilities

def plot_actual_vs_predicted(target, predictions, output_path="output/", filename="keras_regression_actual_vs_predicted.png"):
    # Make predictions
    plt.figure(figsize=(8,8))
    plt.scatter(target, predictions, alpha=0.5)
    plt.xlabel("Actual Strength")
    plt.ylabel("Predicted Strength")
    plt.title("Actual vs Predicted Concrete Strength")
    plt.plot([target.min(), target.max()],
            [target.min(), target.max()], 'r--')
    plt.savefig(output_path+filename)  # Save the plot as a PNG file

def plot_training_validation_loss(history,
                                  output_path=None,
                                  filename="training_validation_loss.png"):
    """
    Plots the training and validation loss over epochs.
    
    history: Keras History object returned by model.fit()
    """
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path+filename) if output_path else plt.show()

def plot_error_over_epochs(error_list, output_path=None):
    """
    Plots the error over epochs.
    
    error_list: list of error values recorded at each epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(error_list)), error_list, color='blue')
    plt.title('Error over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig(output_path+"plot_error_over_epochs.png") if output_path else plt.show()

def plot_activation_functions(z = np.linspace(-10, 10, 400),
                               output_path=None):
    # Plot the activation functions
    plt.figure(figsize=(12, 6))
    sigmoid_grad = analysis_utilities.sigmoid_derivative(z)
    relu_grad = analysis_utilities.relu_derivative(z)
    sigmoid = analysis_utilities.sigmoid(z)
    relu = analysis_utilities.relu(z)
    # Plot Sigmoid and its derivative
    plt.subplot(1, 2, 1)
    plt.plot(z, sigmoid, label='Sigmoid Activation', color='b')
    plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
    plt.title('Sigmoid Activation & Gradient')
    plt.xlabel('Input Value (z)')
    plt.ylabel('Activation / Gradient')
    plt.legend()

    # Plot ReLU and its derivative
    plt.subplot(1, 2, 2)
    plt.plot(z, relu, label='ReLU Activation', color='g')
    plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
    plt.title('ReLU Activation & Gradient')
    plt.xlabel('Input Value (z)')
    plt.ylabel('Activation / Gradient')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path+"activation_functions.png") if output_path else plt.show()

def draw_neural_net(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    """
    Draws a fully connected feedforward neural network.
    
    num_inputs: int, number of input neurons
    num_hidden_layers: int, number of hidden layers
    num_nodes_hidden: list, number of nodes in each hidden layer
    num_nodes_output: int, number of output neurons
    """
    # Network structure
    layers = [num_inputs] + num_nodes_hidden + [num_nodes_output]

    # Figure setup
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    # Neuron positions
    v_spacing = 1
    h_spacing = 2
    neuron_positions = []
    for i, n in enumerate(layers):
        layer_positions = []
        for j in range(n):
            x = i * h_spacing
            y = j * v_spacing - (n - 1) * v_spacing / 2
            layer_positions.append((x, y))
            ax.add_patch(plt.Circle((x, y), 0.1, fill=True, color='skyblue'))
        neuron_positions.append(layer_positions)

    # Draw connections
    for i in range(len(layers) - 1):
        for (x1, y1) in neuron_positions[i]:
            for (x2, y2) in neuron_positions[i + 1]:
                ax.plot([x1, x2], [y1, y2], 'k')

    # Labels
    ax.text(-0.5, 0, 'Input Layer', fontsize=12)

    for i in range(num_hidden_layers):
        ax.text(i* h_spacing + 0.5, 0, f'Hidden {i+1}', fontsize=10)
    
    ax.text((num_hidden_layers+1)*h_spacing - 0.5, 0, 'Output Layer', fontsize=12)

    #plt.show()    