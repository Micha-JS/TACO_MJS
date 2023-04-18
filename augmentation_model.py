import os
import numpy as np
import pandas as pd
import json


import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2B1

import albumentations as A

import mlflow
import mlflow.tensorflow



# Load the configurations

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
dynamic_lr = config['dynamic_lr']

freeze_layers = config['freeze_layers']

seed = config['seed']

  

mlflow.end_run()
mlflow.start_run()


labels = list(train_df.columns)[1:]




# Define data generators for train and validation sets
augmentation_pipeline = A.Compose([
    # Add your desired augmentations here
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p = 0.3),
        A.GridDistortion (num_steps=5, distort_limit=0.6, interpolation=1, border_mode=4, value=None, mask_value=None, normalized=False, p = 0.2),
        A.HorizontalFlip(p = 0.4),
        A.VerticalFlip(p = 0.4),
        A.ToGray(p=0.01),
        A.GaussNoise (var_limit=(0.0024, 0.012), mean=0, per_channel=True, always_apply=False, p = 0.4),
        A.Rotate(limit=350, p = 0.5),
        A.Transpose(p = 0.4),
        A.PixelDropout (dropout_prob=0.01, per_channel=True, drop_value=0, p=0.4),
        A.HueSaturationValue (hue_shift_limit=1, sat_shift_limit=1.1, val_shift_limit=0.1, p = 0.2),
        A.ChannelShuffle(p=0.3)
])

def augment_images(images):
    images = images.astype(np.float32) / 255.0
    augmented_image = augmentation_pipeline(image=images)['image']
    return augmented_image


datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=augment_images
)

train_generator = datagen_train.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='file_name',
    y_col=labels,
    class_mode='raw',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    workers=2,
    multiprocessing=False
)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

val_generator = datagen_val.flow_from_dataframe(
    dataframe=val_df,
    directory=image_dir,
    x_col='file_name',
    y_col=labels,
    class_mode='raw',
    target_size=(img_size, img_size), 
    batch_size=batch_size,
    shuffle=False,
    workers=2,
    multiprocessing=False
)

steps_train = round(train_generator.n / batch_size)
steps_val = round(val_generator.n / batch_size)

def lr_function(epoch):
    start_lr = 1e-6; min_lr = 1e-6; max_lr = 1e-4
    rampup_epochs = 3; sustain_epochs = 0; exp_decay = .8
    
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


mlflow.end_run()
mlflow.start_run()
base_model = EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))


if freeze_layers > 0:
    num_layers = len(base_model.layers)
    index = int(num_layers * freeze_layers)
    
    for i, layer in enumerate(base_model.layers):
        if i < index:
            layer.trainable = False
        else:
            layer.trainable = True
else:
    for layer in base_model.layers:
        layer.trainable = True

frozen_layers = 0
for layer in base_model.layers:
    if not layer.trainable:
        frozen_layers += 1

print("Number of frozen layers in model: ", frozen_layers)
total_layers = len(base_model.layers)
print("Total number of layers in model: ", total_layers)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = Dense(len(labels), activation='sigmoid')(x)


model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

if dynamic_lr == True:
    optimizer = Adam(learning_rate=lr_function(epochs))
else:
    optimizer = Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer, 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'],
                    )

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience, monitor="val_loss", restore_best_weights=True)

class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric('loss', logs['loss'], step=epoch)
        mlflow.log_metric('accuracy', logs['accuracy'], step=epoch)
        mlflow.log_metric('val_loss', logs['val_loss'], step=epoch)
        mlflow.log_metric('val_accuracy', logs['val_accuracy'], step=epoch)
        

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping_cb,MetricsLoggerCallback()],
                    class_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=steps_train,
                    validation_steps=steps_val,
                    max_queue_size=2,
                    workers=4,
                    use_multiprocessing=False
                    )



mlflow.log_param('lr', learning_rate)
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('epochs', epochs)
mlflow.log_param('img_size', img_size)
mlflow.keras.log_model(model, 'model')


mlflow.end_run()