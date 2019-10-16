from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import functools
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

epochs = 1
verbose = 1
batch_size = 100
n = 1

data_set = "oxford_flowers102"

scores = []

top_5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

top_5_acc.__name__ = 'top_5_acc'


def load_data():
    data_train, info = tfds.load(name=data_set, split='test', with_info=True)
    data_test = tfds.load(name=data_set, split='train')
    val_data = tfds.load(name=data_set, split='validation')
    assert isinstance(data_train, tf.data.Dataset)
    assert isinstance(data_test, tf.data.Dataset)
    print(info)

    train_images = []
    train_labels = []

    for example in data_train.take(6149):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        train_images.append(image.numpy())
        train_labels.append(label.numpy())
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = []
    test_labels = []

    for example in data_test.take(1020):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        test_images.append(image.numpy())
        test_labels.append(label.numpy())
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    val_images = []
    val_labels = []

    for example in val_data.take(1020):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        val_images.append(image.numpy())
        val_labels.append(label.numpy())
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    return data_train, data_test,\
           train_images, train_labels,\
           test_images, test_labels,\
           val_images, val_labels,\
           info


def visualize(data_train, data_test, info):
    tfds.show_examples(info, data_train)
    tfds.show_examples(info, data_test)


def run_training(train_images, train_labels, test_images, test_labels, val_images, val_labels):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(input_shape=(256, 256, 3),
                                  kernel_size=(11, 11),
                                  strides=(4, 4),
                                  padding="valid",
                                  filters=3,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))

    model.add(keras.layers.Conv2D(kernel_size=(11, 11),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=256,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))

    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=384,
                                  activation=tf.nn.relu))

    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=384,
                                  activation=tf.nn.relu))

    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=256,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(units=4096,
                                 input_shape=(256, 256, 3),
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))

    model.add(keras.layers.Dense(units=4096,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))

    model.add(keras.layers.Dense(units=102,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))

    model.add(keras.layers.Dense(units=102,
                                 activation=tf.nn.softmax))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc', top_5_acc])

    history = model.fit(train_images,
                        train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_images, val_labels),
                        verbose=1)

    _, accuracy = model.evaluate(test_images,
                                 test_labels,
                                 batch_size=batch_size,
                                 verbose=2)

    scores.append(accuracy)

    history_dict = history.history

    acc = history_dict['acc']
    loss = history_dict['loss']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']

    return model

def predictions(model, test_images, test_labels):

    predictions = model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])


def run_experiment(n):
    for experiments in range(n):
        data_train, data_test, train_images, train_labels, \
        test_images, test_labels, val_images, val_labels, info = load_data()
        visualize(data_train, data_test, info)
        plt.imshow(train_images[0])
        plt.show()
        model = run_training(train_images, train_labels, test_images, test_labels, val_images, val_labels)
        predictions(model, test_images, test_labels)
        plt.tight_layout()
        plt.show()

    print(scores)
    print('M={}'.format(np.mean(scores)), 'STD={}'.format(np.std(scores)))
    print('Min={}'.format(np.min(scores)), 'Max={}'.format(np.max(scores)))


run_experiment(n)

