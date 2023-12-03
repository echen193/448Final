import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocessing: 
# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple RNN model
rnn_model = models.Sequential()

# Reshape input data to fit the RNN
rnn_model.add(layers.Reshape((32, 32 * 3), input_shape=(32, 32, 3)))

# Add an LSTM layer
rnn_model.add(layers.LSTM(128))
rnn_model.add(layers.Dropout(0.2))
# Output layer
rnn_model.add(layers.Flatten())
rnn_model.add(layers.Dense(128, activation='relu'))
rnn_model.add(layers.Dense(10, activation='softmax'))

# Compile the rnn_model
rnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
rnn_model.fit(x_train, y_train, epochs=10,batch_size=32, validation_data=(x_test, y_test))