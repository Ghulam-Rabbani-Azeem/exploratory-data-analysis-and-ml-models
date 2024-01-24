import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Data normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encoding of labels
number_cat = 10
y_train = to_categorical(y_train, number_cat)
y_test = to_categorical(y_test, number_cat)

# CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.3))

cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())

cnn_model.add(Dense(units=512, activation='relu'))
cnn_model.add(Dense(units=512, activation='relu'))
cnn_model.add(Dense(number_cat, activation='softmax'))

# Compile the model
cnn_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# Train the model
history = cnn_model.fit(x_train, y_train, batch_size=64, epochs=1, shuffle=True)

# Evaluate the model
model_evaluation = cnn_model.evaluate(x_test, y_test)
print(f'Accuracy: {model_evaluation[1]}')

# Predict classes
predicted_classes = cnn_model.predict(x_test)
predicted_classes = np.argmax(predicted_classes, axis=1)
y_test = np.argmax(y_test, axis=1)

# Visualization of predictions
l_grid, w_grid = 7, 7
fig, axes = plt.subplots(l_grid, w_grid, figsize=(12, 12))
axes = axes.ravel()
for i in np.arange(0, l_grid * w_grid):
    index = np.random.randint(0, len(x_test))
    axes[i].imshow(x_test[index])
    axes[i].set_title(f'Prediction={predicted_classes[index]}\nTrue={y_test[index]}')
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)

# Confusion matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True)

# Save the model
import os
directory = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')
cnn_model.save(model_path)
