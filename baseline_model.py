import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2B1

import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split
from sklearn import metrics




with open('./config.json', 'r') as f:
    config = json.loads(f.read())

image_dir = config['img_dir']
img_size = config['img_size']

train_df = pd.read_csv(config['train_metadata_filepath'])
val_df = pd.read_csv(config['val_metadata_filepath'])


patience = config['patience']
epochs = config['epochs']

batch_size = config['batch_size'] 
learning_rate = config['learning_rate'] 

seed = config['seed']


mlflow.start_run()



labels = list(train_df.columns)[1:]


# Define data generators for train and validation sets
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    #rotation_range=15,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.005,
    #zoom_range=[0.9, 1.4],
    #horizontal_flip=True,
    #vertical_flip=False,
    #brightness_range=(.8,1.2),
    fill_mode='nearest'
)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_generator = datagen_train.flow_from_dataframe(
    dataframe=train_df,
    directory='/Users/mjs/Desktop/Dev/TACO/data/all_images',
    x_col='file_name',
    y_col=labels,
    class_mode='raw',
    target_size=(img_size, img_size), 
    batch_size=batch_size,
    shuffle=True
)

val_generator = datagen_val.flow_from_dataframe(
    dataframe=val_df,
    directory='/Users/mjs/Desktop/Dev/TACO/data/all_images',
    x_col='file_name',
    y_col=labels,
    class_mode='raw',
    target_size=(img_size, img_size), 
    batch_size=batch_size,
    shuffle=False
)



def lr_function(epoch):
    start_lr = 1e-6; min_lr = 1e-6; max_lr = 1e-4
    rampup_epochs = 5; sustain_epochs = 0; exp_decay = .8
    
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, 
           sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) / rampup_epochs 
                        * epoch + start_lr)
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = ((max_lr - min_lr) * 
                      exp_decay**(epoch - rampup_epochs -
                                    sustain_epochs) + min_lr)
        return lr

    return lr(epoch, start_lr, min_lr, max_lr, 
              rampup_epochs, sustain_epochs, exp_decay)



base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))


for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = Dense(len(labels), activation='sigmoid')(x)


model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer, 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'],
                    )

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)

mlflow.tensorflow.autolog()

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping_cb],
                    )



mlflow.log_param('lr', learning_rate)
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('epochs', epochs)
mlflow.log_param('img_size', img_size)
mlflow.keras.log_model(model, 'model')


mlflow.end_run()


