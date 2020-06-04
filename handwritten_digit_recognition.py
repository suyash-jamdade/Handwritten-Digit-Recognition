import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0

test_images = test_images.reshape(10000, 28, 28, 1) / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape = (28 ,28,1)))
model.add(tf.keras.layers.MaxPool2D(padding='same'))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPool2D(padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_images , training_labels , epochs = 20 )
print(history.epoch, history.history['accuracy'][-1])

val_loss, val_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)

print(len(prediction))

print("The number on the picture is {} ".format(np.argmax(prediction[66])))





