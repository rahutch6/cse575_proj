import tensorflow as tf 
import pathlib

from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 
img_dir = '.\\image_recog_src\\train_validate' 
img_dir = pathlib.Path(img_dir)
image_count = len(list(img_dir.glob('*/*.jpg'))) 

# Training split 
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
sparse_tr_ds = tf.keras.utils.image_dataset_from_directory( 
  pathlib.Path('.\\image_recog_src\\train_validate_sparse'),
  validation_split=0.2, 
  subset="training", 
  seed=150, 
  image_size=(256,256), 
  batch_size=32)
sparse_val_ds = tf.keras.utils.image_dataset_from_directory( 
  pathlib.Path('.\\image_recog_src\\train_validate_sparse'),
  validation_split=0.2, 
  subset="validation", 
  seed=150, 
  image_size=(256,256), 
  batch_size=32)
gen_tr_ds = tf.keras.utils.image_dataset_from_directory( 
  pathlib.Path('.\\image_recog_src\\train_validate_generated'),
  validation_split=0.2, 
  subset="training", 
  seed=150, 
  image_size=(256,256), 
  batch_size=32)
gen_val_ds = tf.keras.utils.image_dataset_from_directory( 
  pathlib.Path('.\\image_recog_src\\train_validate_generated'),
  validation_split=0.2, 
  subset="validation", 
  seed=150, 
  image_size=(256,256), 
  batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(
  pathlib.Path('.\\image_recog_src\\test'),
  image_size=(256, 256),
  batch_size=32,
  labels="inferred"
)