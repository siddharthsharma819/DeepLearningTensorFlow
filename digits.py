import tensorflow as tf
from tensorflow import keras

#  Network and training parameters
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

RESHAPED = 784

x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train_samples')
print(x_test.shape[0], 'test_samples')
print(y_train.shape[0], 'train_value')
print(y_test.shape[0], 'test_value')

y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES,
                             input_shape=(RESHAPED,),
                             name="dense_layer",
                             activation='softmax'))
model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)