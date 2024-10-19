import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
import os
import argparse

from data import load_samples, train_data_generator, validation_data_generator
from model import LPQ_net
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


batch_size = 32
learning_rate = 0.01
epochs = 30
num_classes = 2

def main(args):
    # Define image shape based on input type
    img_size = 256
    img_shape = (img_size, img_size, 1) if args.image_type == 'grayscale' else (img_size, img_size, 3)
    batch_size = 32

    preprocess = 1 if args.image_type == 'grayscale' else 0

    # Load and preprocess the data
    train_data_path = './csv/train.csv'     # Path to the training data
    validation_data_path='./csv/validation.csv' # Path to the validation data


    train_samples = load_samples(train_data_path)
    validation_samples = load_samples(validation_data_path)



    print(len(train_samples))
    print(len(validation_samples))



    train_generator=train_data_generator(train_samples,batch_size,preprocess= preprocess,img_size=img_size) 
    validation_generator = validation_data_generator(validation_samples,batch_size,preprocess = preprocess,img_size=img_size)

    # Create and compile the model
    model = LPQ_net(in_shape=img_shape, num_classes=num_classes)

    opti=Adagrad(learning_rate=learning_rate,decay=(learning_rate/epochs))

    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])

    # Train the model
    if args.image_type == 'grayscale':
        model_filepath = './models/LPQ_NET_Gray.keras'  # Path to save the model for grayscale images
    else:
        model_filepath = './models/LPQ_NET_Color.keras'    # Path to save the model

    checkpoint=ModelCheckpoint(model_filepath,monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')

    history=model.fit(train_generator, 
                    epochs=epochs, 
                    steps_per_epoch=len(train_samples)//batch_size, 
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)//batch_size, 
                    callbacks=[checkpoint])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPQ-NET For Deepfake Detection')
    parser.add_argument('--image_type', type=str, default='color', required=False, help="Image type: 'grayscale' or 'color'")
    # parser.add_argument('--train_dir', type=str, required=True, help='Directory for training images')


    args = parser.parse_args()
    main(args)


