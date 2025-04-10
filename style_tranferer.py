#############################
#      Style Transferer     #
#---------------------------#
# @author: alcoope8@asu.edu #
# @author: rahutch6@asu.edu #
#############################

# Imports #
import numpy as np

# Image processing
from PIL import Image                         # Image processing

# Torch
import torch
import torchvision.models as models

# Scipy
from scipy.optimize import fmin_l_bfgs_b      # Minimization function

# Imageio
import imageio
import pprint

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

  # Normalize the RGB values of each image
  tc_arr    = normalize_rgb(tc_arr, tc_rgb)
  ts_arr    = normalize_rgb(ts_arr, ts_rgb)

  # Create Torch Tensors
  input_shape   = (width, height, 3)
  c_img         = torch.from_numpy(tc_arr.copy())   # Content Image Tensor
  s_img         = torch.from_numpy(ts_arr.copy())   # Style Image Tensor
  combo_img     = torch.empty_like(c_img)           # Combined Image Tensor
  # loss  = Input(shape=(1,))

  # Instantiate the VGG CNN Model
  model_outputs = {} # Dict to store intermediate level outputs
  vgg16         = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
  vgg16.eval()

  # Define a hook to get the output of the 8th layer
  def hook(module, input, output):
    ''' Hook function to get the output of the 8th layer (4th ReLU) '''
    model_outputs['block2_conv2'] = output

  # Register the hook
  b2_c2_hook  = vgg16.features[8].register_forward_hook(hook)

  # Poke the model with a dummy input to get something populated in the dict
  dummy_input = torch.randn(3, 3, 512, 512)
  _ = vgg16(dummy_input)
  # print("block2_conv2 activation shape:", model_outputs['block2_conv2'].shape)

  # Input tensor:
  # - Matrix of content image, style image, and combo image along the batch axis (0).
  # - Represents a batch of three images that will be passed thru the VGG16 CNN
  in_tensor = torch.cat([c_img, s_img, combo_img], dim=0)
  # print(in_tensor.shape)

  # WEIGHTS # TODO: TUNE ME
  c_weight = 0.025
  s_weight = 5.0
  total_variation_weight = 1.0

  # print(layers['block2_conv2'])

  # CONTENT LOSS #
  # layer_features    = layers['block2_conv2']
  # content_features  = layer_features[0, :, :, :]
  # combo_features    = layer_features[2, :, :, :]
  # loss += c_weight * content_loss(content_features, combo_features)

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
  Converts images to 4d numpy arrays of the form (batch size, channels, height, width)  
  Batch size will be one since each image is one thing.  
  Also extracts the mean R, G, and B value for the supplied image. 
  Channels will be 3 for R, G, and B  
  @param image: PIL Image object to format.    
  @returns: 4d numpy array, list containing avg r, g, b values
  '''
  # Formatting
  formatted_array = np.asarray(image, dtype='float32')      # (height, width, channels)
  formatted_array = formatted_array.transpose(2, 0, 1)
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
    image_arr[:, i, :, :] -= avg_rgbs[i]

  # Flip image to BGR as in the paper
  # Todo: necessary?
  image_arr = image_arr[:, ::-1, :, :]
  return image_arr

# def content_loss(content, combination):
#   '''
#   Function to calculate the loss wrt the content
#   '''
#   loss =   tf.keras.metrics.Sum().update_state((combination - content) ** 2).result()
#   return loss

if __name__ == "__main__":
  main()