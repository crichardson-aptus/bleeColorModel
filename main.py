import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('colorData.csv')

# Split the data into input and output variables
inputs = data.iloc[:, 0:3].values
outputs = data.iloc[:, 3:13].values

# Define the neural network model
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))

# Compile the model with a mean squared error loss function and an Adam optimizer
model.compile(loss='mse', optimizer='adam')

# Train the model on the input and output data
model.fit(inputs, outputs, epochs=100, batch_size=10)

# Use the trained model to make predictions on new input data
new_inputs = [[51.79, 62.67, 1.84], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
predictions = model.predict(new_inputs)

print(predictions)

# Save the model to a file
model.save('my_model.h5')

