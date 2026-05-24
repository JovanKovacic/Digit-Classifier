import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Model training
model.fit(x_train, y_train, epochs=20, validation_split=0.1, callbacks=[early_stopping])

# Model evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Loss: {loss}")

# Save model
model.save("digits_model.h5")

# Test model on custom images
for x in range(1, 10):
    img = cv.imread(f"{x}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    img = tf.keras.utils.normalize(img, axis=1)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    print(f"Prediction: {predicted_label}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
