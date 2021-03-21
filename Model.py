import numpy as np
import tensorflow as tf

# Load our training data
npz = np.load('Business_case_data_train.npz')

# split it into our inputs and targets
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)
# load our validation and testing data as well
npz = np.load('Business_case_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('Business_case_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

BATCH_SIZE = 30
output_size = 2
hidden_layer_size = 150

# Setup our model with our designated hidden and output layers
model = tf.keras.Sequential([
    # dense is applying output = activation(dot(input,weight) + bias)
    # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation=tf.keras.layers.LeakyReLU()),
    # Second Hidden Layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    # output layer softmax for classification problem
    tf.keras.layers.Dense(output_size, activation='sigmoid')
])

# setup an early stop feature in order to emit overfitting our model
early_stop = tf.keras.callbacks.EarlyStopping(patience=4)

model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
NUM_EPOCHS = 100

# begin training our date
model.fit(train_inputs, train_targets, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
          validation_data=(validation_inputs, validation_targets), validation_steps=10, verbose=2, callbacks=early_stop)

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

# print out our test loss and test accuracy
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
