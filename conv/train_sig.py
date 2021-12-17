import numpy as np
import pathlib
# import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_DIR = 'data/'
trainig_img_dir = DATA_DIR + 'train/'
test_img_dir = DATA_DIR + 'test/'

# Section0: gathering information about the dataset balance
balance_file = list(open(DATA_DIR + 'balance.txt', 'r'))
balance = []
for i in balance_file:
    balance.append(i.split()[0])

# From: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
pos = int(balance[0])
neg = int(balance[1])
total = pos + neg

weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('\nWeight for class 0: {:.2f}'.format(weight_for_0))
print('\nWeight for class 1: {:.2f}'.format(weight_for_1))


# Section1: Loading images from directories for training and test

# ImageDataGenerator class provides a mechanism to load both small and large dataset.
# Instruct ImageDataGenerator to scale to normalize pixel values to range (0, 1)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.)
# Create a training image iterator that will be loaded in a small batchsize. Resize all images to a #standard size.
train_it = datagen.flow_from_directory(trainig_img_dir, batch_size=8, target_size=(90, 90))
# Create a training image iterator that will be loaded in a small batch size. Resize all images to a #standard size.
test_it = datagen.flow_from_directory(test_img_dir, batch_size=8, target_size=(90, 90))

# Lines 22 through 24 are optional to explore your images.
# Notice, next() function call returns both pixel and labels values as numpy arrays.
train_images, train_labels = train_it.next()
test_images, test_labels = test_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (train_images.shape, train_images.min(), train_images.max()))

# Section 2: Build CNN network and train with training dataset.
# You could pass argument parameters to build_cnn() function to set some of the values
# such as number of filters, strides, activation function, number of layers etc.

# Moodelo original do exercício
def build_cnn():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(1, 1), input_shape=(90, 90, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model

# Dicas
# Diminuir taxa de aprendizado. Valor da taxa talvez deva cair conforme o treino progride.

# Build CNN model
model = build_cnn()
# Compile the model with optimizer and loss function
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model. fit_generator() function iteratively loads large number of images in batches
# history = model.fit(XTraining, YTraining, epochs=ts, validation_data=(XValidation,YValidation), shuffle=True)

history = model.fit_generator(train_it, epochs=30, steps_per_epoch=16,
                              validation_data=test_it, validation_steps=8, class_weight=class_weight)

# Section 3: Save the CNN model to disk for later use.
model_path = "trained_model_conv/"
model.save(filepath=model_path)

# Section 4: Display evaluation metrics
print(history.history.keys())
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')

plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
