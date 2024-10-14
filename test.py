import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
import os
import argparse

from data import load_samples, test_data_generator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

batch_size = 32


def main(args):
    # Define image shape based on input type
    img_size = 256
    img_shape = (
        (img_size, img_size, 1)
        if args.image_type == "grayscale"
        else (img_size, img_size, 3)
    )

    preprocess = 1 if args.image_type == "grayscale" else 0
    batch_size = 32
    test_data_path = "./test.csv"  # Path to the training data

    test_samples = load_samples(test_data_path)

    print(len(test_samples))

    test_generator = test_data_generator(
        test_samples, batch_size, preprocess=preprocess
    )

    # Load the model
    model = (
        tf.keras.models.load_model("./models/LPQ_NET_Gray.keras")
        if args.image_type == "grayscale"
        else tf.keras.models.load_model("./models/LPQ_GRAY.h5")
    )
    batch_size = 32

    # Evaluate the model
    results = model.evaluate(test_generator, batch_size=batch_size, verbose=1)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPQ-NET For Deepfake Detection')
    parser.add_argument('--image_type', type=str, default='color', required=False, help="Image type: 'grayscale' or 'color'")
    # parser.add_argument('--train_dir', type=str, required=True, help='Directory for training images')


    args = parser.parse_args()
    main(args)
