
"""
Author: Henry Powell

Institution: Institute of Neuroscience, Glasgow University, Scotland.

Implementation of AlexNet using Keras with Tensorflow backend. Code will preload the Oxford_Flowers102 dataset.
Learning tracks the model's accuracy, loss, and top 5 error rate. For true comparision of performance to the original
model (Krizhevsky et al. 2010) this implementation will need to be trained on Imagenet2010.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import utils
import functools
import numpy as np
import matplotlib.pyplot as plt

# Set global variables.
epochs = 1
verbose = 1
steps_per_epoch = 10
n = 1

# Set the dataset which will be downladed and stored in system memory.
data_set = "oxford_flowers102"

# This list keeps track of the training metrics for each iteration of training if you require to run an experiment.
# I.e. if you want to train the network n times and see how the training differs between each iteration of training.
acc_scores = []
loss_scores = []
top_5_acc_scores = []

# Set up the top 5 error rate metric.
top_5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top_5_acc.__name__ = 'top_5_acc'


def load_data():
    """
    Function for loading and augmenting the training, testing, and validation data.
    :return: images and labels as numpy arrays (the labels will be one-hot encoded) as well as an info object
    containg information about the loaded dataset.
    """
    # Load the data using TensorFlow datasets API.
    data_train, info = tfds.load(name=data_set, split='test', with_info=True)
    data_test = tfds.load(name=data_set, split='train')
    val_data = tfds.load(name=data_set, split='validation')

    # Ensure that loaded data is of the right type.
    assert isinstance(data_train, tf.data.Dataset)
    assert isinstance(data_test, tf.data.Dataset)

    # Prints the dataset information.
    print(info)

    train_images = []
    train_labels = []

    # Here we take all the samples in the training set (6149), convert the data type to float32 and resize.
    # Since the images in Oxford_Flowers are not pre processed we need to resize them all so that the network
    # takes inputs that are all the same size.
    for example in data_train.take(-1):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        train_images.append(image.numpy())
        train_labels.append(label.numpy())

    # We then convert the lists of images and labels to numpy arrays.
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    # And change the labels to one-hot encoded vectors (this is so we can use the categorical_cross entropy loss
    # function.
    train_labels = utils.to_categorical(train_labels)

    # We now do as above but with the test and validation datasets.
    test_images = []
    test_labels = []

    for example in data_test.take(-1):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        test_images.append(image.numpy())
        test_labels.append(label.numpy())
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    test_labels = utils.to_categorical(test_labels)

    val_images = []
    val_labels = []
    for example in val_data.take(-1):
        image, label = example['image'], example['label']
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        val_images.append(image.numpy())
        val_labels.append(label.numpy())
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    val_labels = utils.to_categorical(val_labels)

    return data_train, data_test,\
           train_images, train_labels,\
           test_images, test_labels,\
           val_images, val_labels,\
           info


def visualize(data_train, data_test, info):
    """
    Short function that visualizes the data set giving 9 samples from each of the training and test datasets
    and their respective labels.
    :param data_train: A tf.data.Dataset object containing the training data
    :param data_test: A tf.data.Dataset object containing the test data
    :param info: dataset.info for getting information about the dataset (number of classes, samples etc.)
    :return: n/a
    """
    tfds.show_examples(info, data_train)
    tfds.show_examples(info, data_test)


def run_training(train_data, test_data, val_data):
    """
    Build, compile, fit, and evaluate the AlexNet model using Keras.
    :param train_data: a tf.data.Dataset object containing (image, label) tuples of training data.
    :param test_data: a tf.data.Dataset object containing (image, label) tuples of test data.
    :param val_data: a tf.data.Dataset object containing (image, label) tuples of validation data.
    :return: trained model object.
    """

    # Set up the sequential model
    model = keras.Sequential()

    # First layer: Convolutional layer with max pooling and batch normalization.
    model.add(keras.layers.Conv2D(input_shape=(256, 256, 3),
                                  kernel_size=(11, 11),
                                  strides=(4, 4),
                                  padding="valid",
                                  filters=3,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.BatchNormalization())

    # Second layer: Convolutional layer with max pooling and batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(11, 11),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=256,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.BatchNormalization())

    # Third layer: Convolutional layer with batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=384,
                                  activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())

    # Fourth layer: Convolutional layer with batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=384,
                                  activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())

    # Fifth layer: Convolutional layer with max pooling and batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=256,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output to feed it to dense layers
    model.add(keras.layers.Flatten())

    # Sixth layer: fully connected layer with 4096 neurons with 40% dropout and batch normalization.
    model.add(keras.layers.Dense(units=4096,
                                 input_shape=(256, 256, 3),
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.BatchNormalization())

    # Seventh layer: fully connected layer with 4096 neurons with 40% dropout and batch normalization.
    model.add(keras.layers.Dense(units=4096,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.BatchNormalization())

    # Eigth layer: fully connected layer with 1000 neurons with 40% dropout and batch normalization.
    model.add(keras.layers.Dense(units=1000,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.BatchNormalization())

    # Output layer: softmax function of 102 classes of the dataset. This integer should be changed to match
    # the number of classes in your dataset if you change from Oxford_Flowers.
    model.add(keras.layers.Dense(units=102,
                                 activation=tf.nn.softmax))

    model.summary()

    # Compile the model using Adam optimize and categorical cross entropy loss function. If your data is not one-hot
    # encoded, change the loss to "sparse_categorical_crossentropy" which accepts integer valued labels rather than
    # 1-0 arrays.
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc', top_5_acc])

    # Fir the model on the training data and validate on the validation data.
    model.fit(train_data,
              epochs=epochs,
              validation_data=val_data,
              verbose=verbose,
              steps_per_epoch=steps_per_epoch)

    print(model.metrics_names)

    # Evaluate the model
    loss, accuracy, top_5 = model.evaluate(test_data,
                                           verbose=verbose,
                                           steps=5)

    # Append the metrics to the scores lists in case you are performing an experiment which involves comparing
    # training over many iterations.
    loss_scores.append(loss)
    acc_scores.append(accuracy)
    top_5_acc_scores.append(top_5)

    return model


def predictions(model, test_images, test_labels):
    """
    Display some examples of the predicions that the network is making on the testing data.
    :param model: model object
    :param test_images: tf.data.Dataset object containing the training data
    :param test_labels: tf.data.Dataset object containing the testing data
    :return: n/a
    """

    predictions = model.predict(test_images)

    plt.subplot(1, 2, 1)
    # Plot first predicted image
    plt.imshow(test_images[0])
    plt.show()

    plt.subplot(1, 2, 2)
    # Plot bar plot of confidence of predictions of possible classes for the first image in the test data
    plt.bar([i for i in range(len(predictions[0]))], predictions[0])
    plt.show()


def run_experiment(n):
    """
    Run an experiment. One experiment loads the dataset, trains the model, and outputs the evaluation metrics after
    training.
    :param n: Number of experiments to perform
    :return: n/a
    """
    for experiments in range(n):

        data_train, data_test, train_images, train_labels, test_images, test_labels, val_images, val_labels, info \
            = load_data()

        visualize(data_train, data_test, info)

        # Print the first resized training image as a sanity check
        plt.imshow(train_images[0])
        plt.show()

        # Make image, label paris into a tf.data.Dataset, shuffle the data and specify batch size.
        train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_data = train_data.repeat().shuffle(1024).batch(100)

        test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_data = test_data.repeat().shuffle(1024).batch(100)

        val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_data = val_data.batch(100)

        model = run_training(train_data, test_data, val_data)

        predictions(model, test_images, test_labels)

    # Print the mean, std, min, and max of the validation accuracy scores from your experiment.
    print(acc_scores)
    print('Mean_accuracy={}'.format(np.mean(acc_scores)), 'STD_accuracy={}'.format(np.std(acc_scores)))
    print('Min_accuracy={}'.format(np.min(acc_scores)), 'Max_accuracy={}'.format(np.max(acc_scores)))


run_experiment(n)

