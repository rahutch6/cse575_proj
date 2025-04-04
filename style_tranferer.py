###################
# Style Transferer
# TODO: Document
##################

# Imports #
import numpy as np

# Image processing
from PIL import Image                         # Image processing

# Keras
import tensorflow as tf
from keras                    import backend
from keras.models             import Model
from keras.applications.vgg16 import VGG16    # Image classification CNN

# Scipy
from scipy.optimize import fmin_l_bfgs_b      # Minimization function

# Imageio
import imageio

# Demo
def main():

  # Useful Constants
  width   = 512
  height  = 512

  # Get the images
  test_content_image  = get_image("image_src/content/guy_at_lake.jpg", width, height)
  test_style_image    = get_image("image_src/style/prettyFlowers.jpg", width, height)
  # save_image(test_style_image, "tst_style")

  # Convert the images to 4D arrays for the CNN to use
  tc_arr, tc_rgb = img_2_arr(test_content_image)
  ts_arr, ts_rgb = img_2_arr(test_style_image)
  # print(tc_arr.shape)
  # print(ts_arr.shape)

  # Normalize the RGB values of each image
  tc_arr    = normalize_rgb(tc_arr, tc_rgb)
  ts_arr    = normalize_rgb(ts_arr, ts_rgb)

  # Create Keras variables
  c_img     = tf.Variable(tc_arr)  
  s_img     = tf.Variable(ts_arr)  
  combo_img = tf.Variable(np.zeros_like(tc_arr))

  # Input tensor:
  # - Matrix combination of content image, style image, and combo image 
  #   along the batch axis (axis 0).
  # - Represents a batch of three images that will be passed thru the VGG16 CNN
  in_tensor = tf.concat([c_img, s_img, combo_img], axis=0)
  print(in_tensor.shape)

# Helper Functions #
def get_image(image_path, width=512, height=512):
  '''
  Function to retrieve and resize the image located at the filepath.  
  @param image_path: path to desired image  
  @param width: rescaled width of the image  
  @param height: rescaled height of the image  
  @returns: resized image
  '''

  c_image = Image.open(image_path)
  c_image = c_image.resize((width, height))
  return c_image

def save_image(image, filename, filetype="PNG"):
  '''
  Function to save image to image_out as a type.  
  @param image: the image object to save   
  @param filename: name of output file  
  @param filetype: format of image (JPEG, PNG, etc)  
  @returns: nothing
  '''

  out_dir = f"./image_out/{filename}.{filetype.lower()}"
  image.save(out_dir)
  print(f"Image saved to: {out_dir}")

def img_2_arr(image):
  '''
  Function to convert images into a good form for processing.  
  Converts images to 4d numpy arrays of the form (batch size, height, width, channels)  
  Batch size will be one since each image is one thing.  
  Also extracts the mean R, G, and B value for the supplied image. 
  Channels will be 3 for R, G, and B  
  @param image: PIL Image object to format.    
  @returns: 4d numpy array, list containing avg r, g, b values
  '''
  # Formatting
  formatted_array = np.asarray(image, dtype='float32')      # (height, width, channels)
  formatted_array = np.expand_dims(formatted_array, axis=0) # (batch size, height, width, channels)

  # Avg RBG 
  rgb_vals = [[], [], []]
  width, height = image.size
  for y in range(height):
    for x in range(width):
      rgb_vals[0].append(image.getpixel((x,y))[0]) # R
      rgb_vals[1].append(image.getpixel((x,y))[1]) # G
      rgb_vals[2].append(image.getpixel((x,y))[2]) # B
  
  rgb_vals[0] = np.mean(rgb_vals[0]) # Avg R
  rgb_vals[1] = np.mean(rgb_vals[1]) # Avg G
  rgb_vals[2] = np.mean(rgb_vals[2]) # Avg B

  return formatted_array, rgb_vals

def normalize_rgb(image_arr, avg_rgbs):
  '''
  Given an input image (4d np array from img_2_arr),
  Subtract the average R, G, and B values.
  @returns: 4d numpy array
  '''

  # Assumes image_arr has 3 channels
  for i in range(3):
    image_arr[:, :, :, i] -= avg_rgbs[i]

  # Flip image to BGR as in the paper
  # Todo: necessary?
  image_arr = image_arr[:, :, :, ::-1]
  return image_arr

if __name__ == "__main__":
  main()