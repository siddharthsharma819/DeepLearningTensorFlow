import tensorflow as tf
from tensorflow import keras

# Network and training
EPOCHS = 50
BATCH_SIZE = 128
VERBOSE=1
NB_CLASSES=10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

# load mnist dataset
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

RESHAPED = 784

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Input normalization to be written as (0,1)
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train.shape[0])
print(x_test.shape[0])

y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,
                             input_shape=(RESHAPED,),
                             name='hidden_layer1',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
                             name='hidden_layer2',
                             activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES,
                             name='output_layer',
                             activation='softmax'))

# model summary
model.summary()

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Model accuracy: ", test_accuracy)