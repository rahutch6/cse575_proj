'''
Simple wrapper script to run the style transferer script on a directory of images.
Usage: 
  run_style_transferer.py --content_src = <path to content directory> --style_src = <path to style directory>

IMPORTANT: must be exactly as many images in content_src as image_src

For Cuda: conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
'''

import argparse
import subprocess as sp
import sys
import os

def main():
  ''' Run style transferer '''

  parser = argparse.ArgumentParser()
  parser.add_argument('--content_src', '-cs', type=str, default="image_src/content",help="directory of content images")
  parser.add_argument('--style_src'  , '-ss', type=str, default="image_src/style",help="directory of style images")
  parser.add_argument('--content_weight', '-cw' , type=float, default=0.025 , help="content weight")
  parser.add_argument('--style_weight'  , '-sw' , type=float, default=5.0   , help="style weight")
  parser.add_argument('--iterations'    , '-itr', type=int,   default=10    , help="Num iterations to train")
  parser.add_argument('--gpu_enable'    , '-ge', action="store_true"        , help="Add flag to use GPU")
  parser.add_argument('--img_out'       , '-io' , type=str,   default="image_out/out", help="Path to output image")
  parser.add_argument('--var_weight'    , '-vw' , type=float, default=1.0   , help="idk")
  args = parser.parse_args()

  # Verify inputs are good
  if not os.path.exists(args.content_src):
    print("Path to content imagess is invalid. Exiting.")
    sys.exit(1)

  if not os.path.exists(args.style_src):
    print("Path to style images is invalid. Exiting.")
    sys.exit(1)

  out_dir       = "image_out"
  content_paths = []
  style_paths   = []
  script_ver = "./style_transferer.py" if not args.gpu_enable else "./style_transferer_gpu.py"

  for file in os.listdir(args.content_src):
    content_paths.append(os.path.join(args.content_src, file))

  for file in os.listdir(args.style_src):
    style_paths.append(os.path.join(args.style_src, file))

  for idx, content_path in enumerate(content_paths):
    c_name = os.path.splitext(os.path.basename(content_path))[0]
    s_name = os.path.splitext(os.path.basename(style_paths[idx]))[0]
    
    out_name = args.img_out
    sp_args = ['python', script_ver, '-ci', content_path, '-si', style_paths[idx], 
                '-io', out_name, '-itr', str(2), '-cw', str(args.content_weight), '-sw', str(args.style_weight), 
                '-itr', str(args.iterations), '-vw', str(args.var_weight)]
    # sp.run(sp_args, stdout=sp.PIPE, universal_newlines=True, shell=False)
    command = ' '.join(sp_args)  # Ensure sp_args is a space-separated string
    os.system(command)
  print(f"Style Paths: {style_paths}")
  print(f"Content Paths: {content_paths}")


if __name__ == "__main__":
  main()