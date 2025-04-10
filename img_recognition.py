import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
  
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 

import pathlib

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# img_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
img_dir = '.\\img_recog\\flower_photos' 
img_dir = pathlib.Path(img_dir)
image_count = len(list(img_dir.glob('*/*.jpg'))) 
# roses = list(img_dir.glob('roses/*')) 
# img = PIL.Image.open(str(roses[0]))
# img.show()

# # Training split 
train_ds = tf.keras.utils.image_dataset_from_directory( 
	img_dir, 
	validation_split=0.2, 
	subset="training", 
	seed=150, 
	image_size=(256, 256), 
	batch_size=32) 
val_ds = tf.keras.utils.image_dataset_from_directory( 
    img_dir, 
    validation_split=0.2, 
    subset="validation", 
    seed=150, 
    image_size=(256,256), 
    batch_size=32)
class_names = train_ds.class_names 

