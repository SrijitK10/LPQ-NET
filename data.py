import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from preprocessing.lpq import lpq



from sklearn.utils import shuffle



def load_samples(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Extract the necessary columns
    file_names = list(data['Path'])
    labels = list(data['Truth'])

    # Combine file names and labels into samples
    samples = [[file_name, label] for file_name, label in zip(file_names, labels)]

    return samples

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    final = lpq(img, winSize=3, freqestim=1)

    return final

def preprocessing_color(image):


    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    R = image[:, :, 2]
    B = image[:, :, 0]
    CR = image1[:, :, 1]

    RR= lpq(R)
    BB= lpq(B)
    CRCR= lpq(CR)

    processed = np.stack((RR, BB,CRCR))

    final= np.swapaxes(processed, 0, 2)

    return final




def train_data_generator(samples, img_size, batch_size=32, shuffle_data=True, preprocess=0, num_classes=2):
    """

    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        samples = shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset : offset + batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_train = []
            y_train = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_path = batch_sample[0]
                label = batch_sample[1]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                

                if preprocess == 0:
                    img = preprocessing_color(img)
                elif preprocess == 1:
                    img = preprocessing(img)
                

                # Add example to arrays
                X_train.append(img)
                y_train.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

            # The generator-y part: yield the next training batch
            yield X_train, y_train


def validation_data_generator(samples, img_size, batch_size=32, shuffle_data=True, preprocess=0, num_classes=2):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        samples = shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset : offset + batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_validation = []
            y_validation = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_path = batch_sample[0]
                label = batch_sample[1]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))


                if preprocess == 0:
                    img = preprocessing_color(img)
                elif preprocess == 1:
                    img = preprocessing(img)


                # Add example to arrays
                X_validation.append(img)
                y_validation.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_validation = np.array(X_validation)
            # y_validation = np.array(y_validation)
            y_validation = np.array(y_validation)

            # X_vaalidation = tf.reshape(X_validation, (-1, 256, 256, 10))
            # y_validation = tf.reshape(y_validation, (-1, 1))

            y_validation = tf.keras.utils.to_categorical(y_validation, num_classes=num_classes)

            # The generator-y part: yield the next training batch
            yield X_validation, y_validation


def test_data_generator(samples, img_size, batch_size=32, shuffle_data=False, preprocess=0, num_classes=2):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        samples = shuffle(samples)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset : offset + batch_size]

            # Initialise X_train and y_train arrays for this batch
            X_test = []
            y_test = []

            # For each example
            for batch_sample in batch_samples:
                # Load image (X) and label (y)
                img_path = batch_sample[0]
                label = batch_sample[1]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                

                if preprocess == 0:
                    img = preprocessing_color(img)
                elif preprocess == 1:
                    img = preprocessing(img)


                # Add example to arrays
                X_test.append(img)

                y_test.append(label)

            # Make sure they're numpy arrays (as opposed to lists)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

            # The generator-y part: yield the next training batch
            yield X_test, y_test
