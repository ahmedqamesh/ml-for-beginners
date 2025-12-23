import os
# Disable oneDNN optimizations if you don't want the floating-point precision warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter('ignore', FutureWarning)
from main_lib import plotting
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.utils import to_categorical
## Example 2: Simple Dataset
# Create a simple dataset
simple_x_train = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
    [18, 19, 20],
    [21, 22, 23],
    [24, 25, 26],
    [27, 28, 29]
])  # Shape: (10, 3)

# Plot the simple_x_train dataset
for i in range(simple_x_train.shape[1]):  # Iterate over each feature (column)
    plt.plot(range(simple_x_train.shape[0]), simple_x_train[:, i], label=f'Feature {i+1}')

plt.title("Visualization of simple_x_train")
plt.xlabel("Sample Index")
plt.ylabel("Feature Value")
plt.legend()
plt.grid(True)
plt.savefig("output/simple_x_train_plot.png")  # Save the plot as a PNG file

simple_y_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Labels for 10 classes
# convert labels to one-hot encoding
target = to_categorical(simple_y_train)
#print("simple_y_train encoded:\n %s"%(simple_y_train[0:10]))

# Normalize the data # Scale values to the range [0, 1]
predictors_norm = simple_x_train / simple_x_train.max()  
n_cols = predictors_norm.shape[1] # number of predictors (features)
# Define a simple model
model = Sequential([
    Input(shape=(n_cols,)),  # Input shape matches the number of features (3)
    Dense(16, activation='relu'),  # Hidden layer with 16 neurons
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class)
    ])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    predictors_norm,
    target,
    epochs=50,
    batch_size=2,
    validation_split=0.2,
    verbose=1
)
# Evaluate the model
loss, accuracy = model.evaluate(predictors_norm, target, verbose=0)
print(f"Model Loss: {loss:.4f}, Model Accuracy: {accuracy:.44f}")   