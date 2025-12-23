import os
import time
# Disable oneDNN optimizations if you don't want the floating-point precision warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter('ignore', FutureWarning)
from main_lib import plotting
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Input
# define regression model
def regression_model(n_cols=None, h_neurons=[]):
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))  # Input shape matches the number of features
    model.add(Dense(h_neurons[0], activation='relu')) # Hidden layer with 50 neurons
    model.add(Dense(h_neurons[1], activation='relu')) # Output layer with 50 neurons (one for each class)
    model.add(Dense(1)) # Output layer with 1 neuron for regression
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
## Example 1: Concrete Compressive Strength Data Set
# Load data
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)
print("Concrete data shape: ", concrete_data.shape)
print("Concrete data head: \n", concrete_data.head())
#print(concrete_data.describe())
time.sleep(1)
print("1. CLean the data..... ")
concrete_data.isnull().sum()
# Get the column names
concrete_data_columns = concrete_data.columns
#Split data into predictors and target
print("2. Split data into predictors and target..... ")
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#print("predictors head: \n", predictors.head())
#print("target head: \n", target.head())
print("3. Normalize the data..... ")
## Scale values to the range [0, 1]
predictors_norm = (predictors - predictors.mean()) / (predictors.std())
#print("predictors norm head: \n", predictors_norm.head())
n_cols = predictors_norm.shape[1] # number of predictors
print("Number of predictors: ", n_cols)
time.sleep(1)
print("4. Define and fit the model..... ")
# Define the model of two hiddedn layers with 50 neurons each
model = regression_model(n_cols=n_cols, h_neurons=[50, 50])
# fit the model
# train and test the model at the same time using the fit method. 
# leave out 30% of the data for validation.
history= model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=1)
# plot the training and validation loss
#If training loss ↓ but validation loss ↑ → overfitting
#If both losses stay high → underfitting
# If both decrease and stabilize → good generalization
plotting.plot_training_validation_loss(history, output_path="output/", filename="keras_regression_validation_loss.png")
predictions = model.predict(predictors_norm)
plotting.plot_actual_vs_predicted(target, predictions, output_path="output/", filename="keras_regression_actual_vs_predicted.png")