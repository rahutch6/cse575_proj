import os
import numpy as np 
import pathlib
import random
import shutil

img_dir = '.\\image_recog_src\\flower_photos' 
img_dir = pathlib.Path(img_dir)
tst_dir = '.\\image_recog_src\\test\\'
trval_dir = '.\\image_recog_src\\train_validate\\'
print("DO NOT RUN THIS AGAIN UNLESS TEST DIRECTORY AND TRAIN_VALIDATE DIRECTORY ARE EMPTY")
exit()
roses = list(img_dir.glob('roses/*')) 
for i in range(len(roses)):
    inPath = str(img_dir) +'\\roses\\'+ str(roses[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'roses\\'+str(roses[i]).split('\\')[-1]
    if ran<(20/len(roses)):
        outPath = tst_dir+outPath
        shutil.copyfile(inPath, outPath)
    else:
        ran=random.random()
        if ran<(100/len(roses)):
            outPath = trval_dir+outPath
            shutil.copyfile(inPath, outPath)
sunflowers = list(img_dir.glob('sunflowers/*')) 
for i in range(len(sunflowers)):
    inPath = str(img_dir) +'\\sunflowers\\'+ str(sunflowers[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'sunflowers\\'+str(sunflowers[i]).split('\\')[-1]
    if ran<(20/len(sunflowers)):
        outPath = tst_dir+outPath
        shutil.copyfile(inPath, outPath)
    else:
        ran=random.random()
        if ran<(100/len(sunflowers)):
            outPath = trval_dir+outPath
            shutil.copyfile(inPath, outPath)
tulips = list(img_dir.glob('tulips/*')) 
for i in range(len(tulips)):
    inPath = str(img_dir) +'\\tulips\\'+ str(tulips[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'tulips\\'+str(tulips[i]).split('\\')[-1]
    if ran<(20/len(tulips)):
        outPath = tst_dir+outPath
        shutil.copyfile(inPath, outPath)
    else:
        ran=random.random()
        if ran<(100/len(tulips)):
            outPath = trval_dir+outPath
            shutil.copyfile(inPath, outPath)
