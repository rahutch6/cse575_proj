import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
  
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 

import dataset_maker as ds

import pathlib
os.environ['KMP_DUPLICATE_LIB_OK']='True'

img_dir           = '.\\image_recog_src\\train_validate' 
img_dir           = pathlib.Path(img_dir)
image_count       = len(list(img_dir.glob('*/*.jpg'))) 

train_ds          = ds.train_ds
val_ds            = ds.val_ds
sparse_tr_ds      = ds.sparse_tr_ds
sparse_val_ds     = ds.sparse_val_ds
gen_tr_ds         = ds.gen_tr_ds
gen_val_ds        = ds.gen_val_ds
test_ds           = ds.test_ds

num_classes=len(train_ds.class_names)

model = Sequential([ 
  layers.Rescaling(1./255, input_shape=(256,256, 3)), 
  layers.Conv2D(16, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(), 
  layers.Conv2D(32, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(), 
  layers.Conv2D(64, 3, padding='same', activation='relu'), 
  layers.MaxPooling2D(), 
  layers.Flatten(), 
  layers.Dense(128, activation='relu'), 
  layers.Dense(num_classes) 
]) 

epochs=15

print(f'\033[92mtraining the full dataset...\033[0m')
model.compile(optimizer='adam', 
      loss=tf.keras.losses.SparseCategoricalCrossentropy( 
        from_logits=True), 
      metrics=['accuracy']) 

history = model.fit( 
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs,
    verbose=1
) 
print(f'\033[93mtesting the full dataset...\033[0m')

full_result = model.evaluate(test_ds, verbose=2)

#=============================================================================
print(f'\033[92mtraining the sparse dataset...\033[0m')
# model.compile(optimizer='adam', 
#       loss=tf.keras.losses.SparseCategoricalCrossentropy( 
#         from_logits=True), 
#       metrics=['accuracy']) 
keras.backend.clear_session()
history = model.fit( 
    sparse_tr_ds, 
    validation_data=sparse_val_ds, 
    epochs=epochs,
    verbose=1
) 
print(f'\033[93mtesting the sparse dataset...\033[0m')
sparse_result = model.evaluate(test_ds, verbose=2)
#=============================================================================
print(f'\033[92mtraining the generated dataset...\033[0m')
keras.backend.clear_session()
# model.compile(optimizer='adam', 
#       loss=tf.keras.losses.SparseCategoricalCrossentropy( 
#         from_logits=True), 
#       metrics=['accuracy']) 
history = model.fit( 
    gen_tr_ds, 
    validation_data=gen_val_ds, 
    epochs=epochs,
    verbose=1
) 

print(f'\033[93mtesting the generated dataset...\033[0m')
generated_result = model.evaluate(test_ds, verbose=2)

#=============================================================================
print(f'\033[95mfull data set test accuracy = {full_result[1]}\033[0m')
print(f'\033[95msparse data set test accuracy = {sparse_result[1]}\033[0m')
print(f'\033[95mgenerated data set test accuracy = {generated_result[1]}\033[0m')
