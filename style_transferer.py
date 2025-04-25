#############################
#      Style Transferer     #
#---------------------------#
# @author: alcoope8@asu.edu #
# @author: rahutch6@asu.edu #
#############################

# Imports #
import time
import numpy as np
import pprint
import argparse
import os
os.environ["MKL_THREADING_LAYER"]   = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
import sys

# Image processing
from PIL import Image                         # Image processing

# Torch
import torch
import torchvision.models as models

# Scipy
from scipy.optimize import fmin_l_bfgs_b      # Minimization function

# Useful Constants
width     = 512
height    = 512
channels  = 3

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--content_image' , '-ci' , type=str                  , help="Path to content image"  )
  parser.add_argument('--style_image'   , '-si' , type=str                  , help="Path to style image"    )
  parser.add_argument('--img_out'       , '-io' , type=str,   default="image_out/out", help="Path to output image")
  parser.add_argument('--iterations'    , '-itr', type=int,   default=10    , help="Num iterations to train")
  parser.add_argument('--content_weight', '-cw' , type=float, default=0.025 , help="content weight")
  parser.add_argument('--style_weight'  , '-sw' , type=float, default=5.0   , help="style weight")
  parser.add_argument('--var_weight'    , '-vw' , type=float, default=1.0   , help="idk")
  args = parser.parse_args()

  validate_args(args)
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("\n----- Running Style Transfer -----")

  # Get the images
  raw_content_image   = get_image(args.content_image, width, height)
  raw_style_image     = get_image(args.style_image, width, height)
  print("\tAcquired content and style images")
  # save_image(raw_style_image, "tst_style")

  # Convert the images to 4D arrays for the CNN to use
  tc_arr, tc_rgb      = img_2_arr(raw_content_image)
  ts_arr, ts_rgb      = img_2_arr(raw_style_image)

  # Normalize the RGB values of each image
  tc_arr              = normalize_rgb(tc_arr, tc_rgb)
  ts_arr              = normalize_rgb(ts_arr, ts_rgb)

  # Create Torch Tensors
  c_img               = torch.from_numpy(tc_arr.copy())   # Content Image Tensor
  s_img               = torch.from_numpy(ts_arr.copy())   # Style Image Tensor
  combo_img           = torch.empty_like(c_img, requires_grad=True)           # Combined Image Tensor
  loss                = torch.zeros(1)                    # Loss Tensor
  print("\tCreated tensors")
  # Input tensor:
  # - Matrix of content image, style image, and combo image along the batch axis (0).
  # - Represents a batch of three images that will be passed thru the VGG16 CNN
  in_tensor = torch.cat([c_img, s_img, combo_img], dim=0)

  # These are the layers of the CNN we need output from
  layers = {
    'block1_conv2': 3,
    'block2_conv2': 8,
    'block3_conv3': 15,
    'block4_conv3': 22,
    'block5_conv3': 29
  }
  layer_outputs = {} # Dict to store intermediate level outputs

  # Instantiate the VGG CNN Model
  vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
  vgg16.eval()

  # Define a hook to get the output of the layers
  def get_activation(name):
    ''' Function to generate hooks for each layer '''

    def hook(module, input, output):
      ''' Hook function to get the output of each layer '''

      layer_outputs[name] = output
    return hook

  # Register the hook for each relevant layer
  for layer_name, idx in layers.items():
    vgg16.features[idx].register_forward_hook(get_activation(layer_name))
  print("\tRegistered Model Hooks")

  # Send in the input tensor
  vgg16(in_tensor) # ; print("block2_conv2 activation shape:", layer_outputs['block5_conv3'].shape)
  print("\tTensor fed to model")

  # WEIGHTS # TODO: TUNE ME
  c_weight                = args.content_weight
  s_weight                = args.style_weight
  total_variation_weight  = args.var_weight

  # CONTENT LOSS #
  layer_features    = layer_outputs['block2_conv2']
  content_features  = layer_features[0]
  combo_features    = layer_features[2]
  loss += c_weight * content_loss(content_features, combo_features)

  # STYLE LOSS #
  for layer in layers:
    layer_features  = layer_outputs[layer]
    style_features  = layer_features[1]
    combo_features  = layer_features[2]
    s_loss          = style_loss(style_features, combo_features)
    loss           += (s_weight / len(layers)) * s_loss
  
  # VARIATION LOSS #
  loss += total_variation_weight * total_variation_loss(combo_img)
  print("\tLoss Calculated")

  # GRADIENT #
  grads     = torch.autograd.grad(loss, combo_img)[0]
  outputs   = [loss]
  outputs  += grads

  def eval_loss_and_grads(x):
    '''
    Given a flattened numpy array x representing the combination image, update combo_img, 
    run a forward-backward pass through vgg16 to compute loss and gradients,
    and return the loss and flattened gradients as a numpy array.
    '''

    # Reshape x into a tensor of shape (1, 3, height, width)
    x_tensor = torch.from_numpy(x.reshape((1, 3, height, width))).float()

    with torch.no_grad():
        combo_img.copy_(x_tensor)

    if combo_img.grad is not None:
        combo_img.grad.zero_()

    layer_outputs.clear()     # Clear previously stored activations from the hooks

    new_in_tensor = torch.cat([c_img, s_img, combo_img], dim=0)
    vgg16(new_in_tensor)

    current_loss = torch.zeros(1)

    # -- Content Loss from block2_conv2 -- #
    lf = layer_outputs['block2_conv2']
    current_loss += c_weight * content_loss(lf[0], lf[2])

    # -- Style Loss from each selected layer -- #
    for layer in layers:
        lf = layer_outputs[layer]
        current_loss += (s_weight / len(layers)) * style_loss(lf[1], lf[2])

    # -- Total Variation Loss on combo_img -- #
    current_loss += total_variation_weight * total_variation_loss(combo_img)

    # compute gradients for combo_img
    current_loss.backward()
    grad_vals = combo_img.grad.cpu().numpy().flatten().astype('float64')
    return current_loss.item(), grad_vals
  
  # TODO: Document
  class Evaluator(object):

    def __init__(self):
      self.loss_val   = None
      self.grad_vals  = None

    def loss(self, x):
      f_loss_val, f_grad_vals = eval_loss_and_grads(x)
      self.loss_val = f_loss_val
      self.grad_vals = f_grad_vals
      return f_loss_val

    def grads(self, x):
      grad_vals = np.copy(self.grad_vals)
      self.loss_val = None
      self.grad_vals = None
      return grad_vals
  
  # EVALUATION #
  evaluator = Evaluator()
  x = np.random.uniform(0, 255, (1, 3, height, width)) - 128

  for i in range(args.iterations):
    print('\t\tStart of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('\t\t\tCurrent loss value:', min_val)
    end_time = time.time()
    print('\t\t\tIteration %d completed in %ds' % (i, end_time - start_time))

  output_img = inverse_image_transform(x, tc_rgb)
  save_image(output_img, args.img_out)

def total_variation_loss(x):
  '''
  Variation loss is a regularization term that smooths the output.
  Less noisy output.
  '''
  a = (x[:, :, :height-1, :width-1] - x[:, :, 1:, :width-1])  ** 2  # Differences along vertical direction
  b = (x[:, :, :height-1, :width-1] - x[:, :, :height-1, 1:]) ** 2  # Differences along horizontal direction
  return torch.sum((a+b) ** 1.25)

def gram_matrix(x):
  '''
  Function to compute the gram matrix.
  Terms are proprtional to the covariances of corresponding features.
  Extract style independent of the content -- spacially irrelevant statistics
  '''
  C, H, W  = x.size()

  # Flatten each image's spatial dims into one dimension
  features    = x.view(C, H * W)

  # Compute batch-wise Gram matrix: for each example, compute (C, H*W) x (H*W, C)
  gram = torch.mm(features, features.t())
  return gram

def style_loss(style, combination):
  '''
  Function to calculate the loss wrt the style
  '''
  s = gram_matrix(style)
  c = gram_matrix(combination)
  size = height * width
  loss = torch.sum((s - c) ** 2) / (4.0 * (channels ** 2) * (size ** 2))
  return loss

def content_loss(content, combination):
  '''
  Function to calculate the loss wrt the content
  '''
  loss = torch.nn.functional.mse_loss(combination, content, reduction="sum")
  return loss

def get_image(image_path, width=512, height=512):
  '''
  Function to retrieve and resize the image located at the filepath.
  @param image_path: path to desired image
  @param width: rescaled width of the image
  @param height: rescaled height of the image
  @returns: resized image
  '''

  c_image = Image.open(image_path)
  if c_image.format != 'JPEG':
    c_image = c_image.convert("RGB")
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

  out_dir = f"{filename}.{filetype.lower()}"
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

  for i in range(3):
    image_arr[:, i, :, :] -= avg_rgbs[i]

  # Flip image to BGR as in the paper
  # Todo: necessary?
  image_arr = image_arr[:, ::-1, :, :]
  return image_arr

def inverse_image_transform(image, avg_rgbs):
    ''' Convert the image in array form back to a PIL Image '''

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    if image.dim() == 1:
        image = image.view(1, 3, height, width)
    elif image.dim() == 4:
        image = image
    else:
        raise ValueError("Unexpected image tensor dimensions")
    
    # Remove the batch dimension
    image = image.squeeze(0)
    
    # Clone the tensor to avoid modifying the original.
    image = image.clone()
    
    # Permute from (C, H, W) to (H, W, C)
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    
    # Re-Reverse channel order (BGR to RGB)
    image = image[:, :, ::-1]
    
    # Add back the mean RGB values
    image[:, :, 0] += avg_rgbs[0]
    image[:, :, 1] += avg_rgbs[1]
    image[:, :, 2] += avg_rgbs[2]
    
    # Clip to valid image range and convert to uint8
    image = np.clip(image, 0, 255).astype('uint8')
    
    return Image.fromarray(image)

def validate_args(args):
  if args.content_image:
    if not os.path.exists(args.content_image):
      print("Path to content image is invalid. Exiting.")
      sys.exit(1)
  else:
    print("Path to content image not provided. Exiting.")
    sys.exit(1)

  if args.style_image:
    if not os.path.exists(args.style_image):
      print("Path to style image is invalid. Exiting.")
      sys.exit(1)
  else:
    print("Path to style image not provided. Exiting.")
    sys.exit(1)

if __name__ == "__main__":
  main()