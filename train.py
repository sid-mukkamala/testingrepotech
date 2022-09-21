from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

from tensorflow.keras import regularizers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import initializers

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt


# Configuration
BATCH_SIZE = 32
TOP_DROP_OUT = 0.1
NUM_CLASSES = 4
INPUT_SHAPE =  (300, 150, 3)
EFF_INPUT_SHAPE = (224, 224, 3)
LAYER_UNFREEZE = 20
AUGMENTATION = {
    'rot': 0.1, 
    'height': 0.05, 
    'width': 0.05,
    'contrast': 0.1
}
EPOCHS_TOP = 10
EPOCHS_ALL = 10


def plot_history(history, metric, tag):
    
  metric_values = history.history[metric]
  val_metric_values = history.history['val_'+metric]
  epochs = range(1, len(metric_values)+1)

  plt.plot(epochs, metric_values, label='Training '+metric)
  plt.xlabel('Epochs')
  plt.ylabel(metric)
  plt.legend()

  plt.plot(epochs, val_metric_values, label='validation '+metric)
  plt.xlabel('Epochs')
  plt.ylabel(metric)
  plt.legend()

  plt.savefig(f'validation/training_{metric}_{tag}.png')
  plt.clf()

def build_model(num_classes, augmentation=AUGMENTATION, weights=None, from_logits=True):

    x = layers.Input(shape=INPUT_SHAPE)

    if augmentation is not None:   
        img_augmentation = Sequential(
            [
                layers.RandomRotation(factor=augmentation['rot']),
                layers.RandomTranslation(height_factor=augmentation['height'], width_factor=augmentation['width']),
                layers.RandomFlip(),
                layers.RandomContrast(factor=augmentation['contrast']),
            ],
        name='img_augmentation',
        )

        x = img_augmentation(x)

    
    model = EfficientNetB0(include_top=False, input_tensor=x, weights=None, input_shape=INPUT_SHAPE)
    model_init = EfficientNetB0(include_top=False, input_shape=INPUT_SHAPE, weights='imagenet')
    model.set_weights(model_init.get_weights())

    # Freeze the pretrained weights
    model.trainable = False

    x = model.output

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(TOP_DROP_OUT, name='top_dropout')(x)

    fan_in = int(x.shape[-1])
    scale = np.sqrt(1 / fan_in) / 6
    x = layers.Dense(num_classes, name='classifier',
            kernel_initializer=initializers.RandomUniform(minval=0, maxval=scale, seed=None),
            bias_initializer=initializers.Constant(value=-4),
            kernel_constraint=NonNeg(),
            activation=None if from_logits else 'sigmoid',
            activity_regularizer=regularizers.l1(0.001))(x)
    
    # Compile
    model = Model(model.inputs, x, name='EfficientNet')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=from_logits),
        metrics=['accuracy']
    )

    return model

def unfreeze_model(model):
    # We unfreeze the top LAYER_UNFREEZE layers while leaving BatchNorm layers frozen
    for layer in model.layers[-LAYER_UNFREEZE:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

dataset_name = 'bee_dataset'
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    dataset_name, split=('train[:50%]', 'train[50%:75%]', 'train[75%:]'), 
    with_info=True, as_supervised=True, shuffle_files=True
)

# Stack targets
def input_preprocess(image, output):
    label = tf.stack([output['cooling_output'],
                      output['pollen_output'], 
                      output['varroa_output'], 
                      output['wasps_output']])
    return image, label


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)

ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(input_preprocess)
ds_val = ds_val.batch(batch_size=BATCH_SIZE, drop_remainder=True)

model = build_model(num_classes=NUM_CLASSES)

model.summary()

# Train top layer
  
hist = model.fit(ds_train, epochs=EPOCHS_TOP, validation_data=ds_val, verbose=2)
plot_history(hist, 'loss', 'top_layer')

model = unfreeze_model(model)

model.summary()

# Train deep layers
hist = model.fit(ds_train, epochs=EPOCHS_ALL, validation_data=ds_val, verbose=2)
plot_history(hist, 'loss', 'all_layers')

# Build model without augmentation layer 
model = Model(model.get_layer('rescaling').input, 
              layers.Activation("sigmoid")(model.output),
              name='EfficientNet')

# Save model
model.training=False
model.save('output/bee')