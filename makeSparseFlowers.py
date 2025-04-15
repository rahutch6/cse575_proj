import os
import numpy as np 
import pathlib
import shutil

img_dir = '.\\image_recog_src\\train_validate\\'
img_dir = pathlib.Path(img_dir)
sparse_dir = '.\\image_recog_src\\train_validate_sparse\\'
gen_dir = '.\\image_recog_src\\train_validate_generated\\'

print("DO NOT RUN THIS AGAIN UNLESS TRAIN_VALIDATE_SPARSE DIRECTORY AND TRAIN_VALIDATE_GENERATED DIRECTORY ARE EMPTY")
exit()
roses = list(img_dir.glob('roses/*'))
for i in range(len(roses)):
    inPath = str(img_dir) +'\\roses\\'+ str(roses[i]).split('\\')[-1]
    outPath = 'roses\\'+str(roses[i]).split('\\')[-1]
    if i%4!=0:
        shutil.copyfile(inPath, sparse_dir + outPath)
        shutil.copyfile(inPath, gen_dir + outPath)
sunflowers = list(img_dir.glob('sunflowers/*'))
for i in range(len(sunflowers)):
    inPath = str(img_dir) +'\\sunflowers\\'+ str(sunflowers[i]).split('\\')[-1]
    outPath = 'sunflowers\\'+str(sunflowers[i]).split('\\')[-1]
    if i%4!=0:
        shutil.copyfile(inPath, sparse_dir + outPath)
        shutil.copyfile(inPath, gen_dir + outPath)
tulips = list(img_dir.glob('tulips/*'))
for i in range(len(tulips)):
    inPath = str(img_dir) +'\\tulips\\'+ str(tulips[i]).split('\\')[-1]
    outPath = 'tulips\\'+str(tulips[i]).split('\\')[-1]
    if i%4!=0:
        shutil.copyfile(inPath, sparse_dir + outPath)
        shutil.copyfile(inPath, gen_dir + outPath)

tulips = list(pathlib.Path(sparse_dir).glob('tulips/*'))
