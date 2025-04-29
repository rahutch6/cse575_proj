import argparse
import os
import numpy as np 
import pathlib
import random
import shutil

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--splitFlowers', '-sp', action="store_true", help="add flag to make 'full' dataset and test data")
  parser.add_argument('--makeSparseFlowers', '-dep', action="store_true",help="add flag to divide data in train_validate into sparse and generated dirs")
  parser.add_argument('--pullStyleImages','-sty',action="store_true",help="add flag to pull images from sparse dir to be the style images and content images")
  parser.add_argument('--generateImages','-gen',action="store_true",help='add flag to generate images')
  parser.add_argument('--content_src', '-cs', type=str, default="image_src/content",help="directory of content images")
  parser.add_argument('--style_src'  , '-ss', type=str, default="image_src/style",help="directory of style images")
  parser.add_argument('--content_weight', '-cw' , type=float, default=0.025 , help="content weight")
  parser.add_argument('--style_weight'  , '-sw' , type=float, default=5.0   , help="style weight")
  parser.add_argument('--iterations'    , '-itr', type=int,   default=10    , help="Num iterations to train")
  parser.add_argument('--var_weight'    , '-vw' , type=float, default=1.0   , help="idk")
  parser.add_argument('--gpu_enable'    , '-ge', action="store_true"        , help="Add flag to use GPU")
  args = parser.parse_args()

  if args.splitFlowers:
    splitFlows()
  if args.makeSparseFlowers:
    depleteFlowers()
  if args.pullStyleImages:
    getStyle()
  if args.generateImages:
    sp_args = ['python', '.\\run_style_transfer.py', '-cw', str(args.content_weight),
               '-sw', str(args.style_weight), '-itr', str(args.iterations), '-vw', str(args.var_weight)]
    if args.gpu_enable: sp_args += ['-ge']

    for i in ['roses', 'sunflowers','tulips']:
    #   new_args=sp_args.copy() + ['-io', f'.\\image_out','-cs', f'{args.content_src}\\{i}', '-ss', f'{args.style_src}\\{i}']
      new_args=sp_args.copy() + ['-io', f'.\\image_recog_src\\train_validate_generated\\{i}','-cs', f'{args.content_src}\\{i}', '-ss', f'{args.style_src}\\{i}']
      command = ' '.join(new_args)  # Ensure sp_args is a space-separated string
      os.system(command)


def splitFlows():
  img_dir = '.\\image_recog_src\\flower_photos' 
  img_dir = pathlib.Path(img_dir)
  tst_dir = '.\\image_recog_src\\test\\'
  trval_dir = '.\\image_recog_src\\train_validate\\'
  roses = list(img_dir.glob('roses/*')) 
  sunflowers = list(img_dir.glob('sunflowers/*')) 
  tulips = list(img_dir.glob('tulips/*')) 

  for i in range(len(roses)):
      inPath = str(img_dir) +'\\roses\\'+ str(roses[i]).split('\\')[-1]
      ran = random.random()
      outPath = 'roses\\'+str(roses[i]).split('\\')[-1]
      if ran<(20/len(roses)):
          outPath = tst_dir+outPath
          shutil.copyfile(inPath, outPath)
      else:
          ran=random.random()
          if ran<(150/len(roses)):
              outPath = trval_dir+outPath
              shutil.copyfile(inPath, outPath)
  for i in range(len(sunflowers)):
      inPath = str(img_dir) +'\\sunflowers\\'+ str(sunflowers[i]).split('\\')[-1]
      ran = random.random()
      outPath = 'sunflowers\\'+str(sunflowers[i]).split('\\')[-1]
      if ran<(20/len(sunflowers)):
          outPath = tst_dir+outPath
          shutil.copyfile(inPath, outPath)
      else:
          ran=random.random()
          if ran<(150/len(sunflowers)):
              outPath = trval_dir+outPath
              shutil.copyfile(inPath, outPath)
  for i in range(len(tulips)):
      inPath = str(img_dir) +'\\tulips\\'+ str(tulips[i]).split('\\')[-1]
      ran = random.random()
      outPath = 'tulips\\'+str(tulips[i]).split('\\')[-1]
      if ran<(20/len(tulips)):
          outPath = tst_dir+outPath
          shutil.copyfile(inPath, outPath)
      else:
          ran=random.random()
          if ran<(150/len(tulips)):
              outPath = trval_dir+outPath
              shutil.copyfile(inPath, outPath)
  print(f'og roses: {len(roses)}\n\ttest roses: {len(list(pathlib.Path(tst_dir).glob("roses/*")))}\n\ttr/val roses: {len(list(pathlib.Path(trval_dir).glob("roses/*")))}')
  print(f'og sunflowers: {len(sunflowers)}\n\ttest sunflowers: {len(list(pathlib.Path(tst_dir).glob("sunflowers/*")))}\n\ttr/val sunflowers: {len(list(pathlib.Path(trval_dir).glob("sunflowers/*")))}')
  print(f'og tulips: {len(tulips)}\n\ttest tulips: {len(list(pathlib.Path(tst_dir).glob("tulips/*")))}\n\ttr/val tulips: {len(list(pathlib.Path(trval_dir).glob("tulips/*")))}')

def depleteFlowers():
  img_dir = '.\\image_recog_src\\train_validate\\'
  img_dir = pathlib.Path(img_dir)
  sparse_dir = '.\\image_recog_src\\train_validate_sparse\\'
  gen_dir = '.\\image_recog_src\\train_validate_generated\\'
  roses = list(img_dir.glob('roses/*'))
  print(f'og roses: {len(roses)}')
  sunflowers = list(img_dir.glob('sunflowers/*'))
  print(f'og sunflowers: {len(sunflowers)}')
  tulips = list(img_dir.glob('tulips/*'))
  print(f'og tulips: {len(tulips)}')

  for i in range(len(roses)):
      inPath = str(img_dir) +'\\roses\\'+ str(roses[i]).split('\\')[-1]
      outPath = 'roses\\'+str(roses[i]).split('\\')[-1]
      if i%3!=0:
          shutil.copyfile(inPath, sparse_dir + outPath)
          shutil.copyfile(inPath, gen_dir + outPath)
  for i in range(len(sunflowers)):
      inPath = str(img_dir) +'\\sunflowers\\'+ str(sunflowers[i]).split('\\')[-1]
      outPath = 'sunflowers\\'+str(sunflowers[i]).split('\\')[-1]
      if i%3!=0:
          shutil.copyfile(inPath, sparse_dir + outPath)
          shutil.copyfile(inPath, gen_dir + outPath)

  for i in range(len(tulips)):
      inPath = str(img_dir) +'\\tulips\\'+ str(tulips[i]).split('\\')[-1]
      outPath = 'tulips\\'+str(tulips[i]).split('\\')[-1]
      if i%3!=0:
          shutil.copyfile(inPath, sparse_dir + outPath)
          shutil.copyfile(inPath, gen_dir + outPath)

  print(f"\tnew roses: {len(list(pathlib.Path(sparse_dir).glob('roses/*')))}")
  print(f"\tnew sunflowers: {len(list(pathlib.Path(sparse_dir).glob('sunflowers/*')))}")
  print(f"\tnew tulips: {len(list(pathlib.Path(sparse_dir).glob('tulips/*')))}")

def getStyle():
  img_dir = '.\\image_recog_src\\train_validate_sparse' 
  img_dir = pathlib.Path(img_dir)
  sty_dir = '.\\image_recog_src\\style'
  ctnt_dir = '.\\image_recog_src\\content'

  roses = list(img_dir.glob('roses/*')) 
  sunflowers = list(img_dir.glob('sunflowers/*')) 
  tulips = list(img_dir.glob('tulips/*')) 

  roseStyle = random.sample(roses,60)
  sunflowerStyle = random.sample(sunflowers,60)
  tulipStyle = random.sample(tulips,60)
  for i in range(60):
    if i%2==0:
      shutil.copyfile(str(roseStyle[i]), str(sty_dir)+'\\roses\\' + str(roseStyle[i]).split('\\')[-1])
      shutil.copyfile(str(sunflowerStyle[i]), str(sty_dir)+'\\sunflowers\\' + str(sunflowerStyle[i]).split('\\')[-1])
      shutil.copyfile(str(tulipStyle[i]), str(sty_dir)+'\\tulips\\' + str(tulipStyle[i]).split('\\')[-1])
    else:
      shutil.copyfile(str(roseStyle[i]), str(ctnt_dir)+'\\roses\\'+ str(roseStyle[i]).split('\\')[-1])
      shutil.copyfile(str(sunflowerStyle[i]), str(ctnt_dir)+'\\sunflowers\\'+ str(sunflowerStyle[i]).split('\\')[-1])
      shutil.copyfile(str(tulipStyle[i]), str(ctnt_dir)+'\\tulips\\'+ str(tulipStyle[i]).split('\\')[-1])
       
  print(f"roses style: {len(list(pathlib.Path(sty_dir).glob('roses/*'))) }")
  print(f"sunflowers style: {len(list(pathlib.Path(sty_dir).glob('sunflowers/*'))) }")
  print(f"tulips style: {len(list(pathlib.Path(sty_dir).glob('tulips/*'))) }")
  print(f"roses content: {len(list(pathlib.Path(ctnt_dir).glob('roses/*'))) }")
  print(f"sunflowers content: {len(list(pathlib.Path(ctnt_dir).glob('sunflowers/*'))) }")
  print(f"tulips content: {len(list(pathlib.Path(ctnt_dir).glob('tulips/*'))) }")


if __name__ == "__main__":
  main()