# Version 1: of modeling with machine learning
# Arnav Singh
# Only using AE, OCB, while predicting OCB location

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('AE_OCB2.csv')

# Assuming columns: 'AE', 'OCB'
# Separate the features and the target variable
X = data[['AE']].values
y = data['OCB'].values

# Normalize the input features
input_scaler = MinMaxScaler()
X = input_scaler.fit_transform(X)

# Normalize the target variable
output_scaler = MinMaxScaler()
y = output_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model, choose 10 epochs, because loss and mae values stabilize after like 2
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test MAE: {test_mae}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions to original scale
predictions_rescaled = output_scaler.inverse_transform(predictions)

# Print the first 10 predictions
print(predictions_rescaled[:10])

# Plot training & validation loss values
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot predictions vs actual values
plt.subplot(1, 2, 2)
plt.scatter(input_scaler.inverse_transform(X_test), output_scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual', marker = '.')
plt.scatter(input_scaler.inverse_transform(X_test), predictions_rescaled, label='Predicted', marker = '.')
plt.title('Actual vs Predicted OCB Location')
plt.xlabel('AE')
plt.ylabel('OCB Location')
plt.legend()

plt.tight_layout()
plt.show()

#Creating Bins for AE
# Ask for user input and display data within the AE range
try:
    user_input = float(input("Enter an AE value to see data within ±10 range: "))
except ValueError:
    print("Please enter a valid numerical AE value.")
    user_input = None

# Check if the user input is valid
if user_input is not None:
    # Create AE bins
    ae_values = input_scaler.inverse_transform(X)[:, 0]

    # Determine the range
    lower_bound = user_input - 10
    upper_bound = user_input + 10

    # Filter the data
    mask = (ae_values >= lower_bound) & (ae_values <= upper_bound)
    filtered_ae = ae_values[mask]
    filtered_ocb = output_scaler.inverse_transform(y.reshape(-1, 1))[mask]

    # Display the filtered data
   # print("\nFiltered AE and OCB Location Data within ±10 range:")
    #for ae, ocb in zip(filtered_ae, filtered_ocb):
     #   print(f"AE: {ae:.2f}, OCB Location: {ocb[0]:.2f}")

    # Plot the filtered data
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_ae, filtered_ocb, color='r', label='Filtered Data', marker='.')
    plt.scatter(ae_values, output_scaler.inverse_transform(y.reshape(-1, 1)), color='b', alpha=0.3, label='All Data',
                marker='.')
    plt.scatter(input_scaler.inverse_transform(X_test), predictions_rescaled, color='g', label='Predicted', marker = '.')
    plt.xlim(lower_bound - 15, upper_bound + 15)  # Zoom in on the specified AE range
    plt.ylim(60, 90)
    plt.title(f'OCB Location for AE values within ±10 of {user_input} with Prediction')
    plt.xlabel('AE')
    plt.ylabel('OCB Location')
    plt.legend()
    plt.show()
else:
    print("No valid input provided.")