import os
import numpy as np 
import pathlib
import random
import shutil

img_dir = '.\\img_recog\\flower_photos' 
img_dir = pathlib.Path(img_dir)
tst_dir = '.\\img_recog\\test\\'
trval_dir = '.\\img_recog\\train_validate\\'
roses = list(img_dir.glob('roses/*')) 
print('DO NOT RUN THIS AGAIN UNLESS YOU HAVE REMOVED THE TEST AND TRAIN_VALIDATE DATA IN THE IMAGE FOLDERS')
exit()
for i in range(len(roses)):
    inPath = str(img_dir) +'\\roses\\'+ str(roses[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'roses\\'+str(roses[i]).split('\\')[-1]
    if ran<.2:
        outPath = tst_dir+outPath
    else:
        outPath = trval_dir+outPath
    shutil.copyfile(inPath, outPath)
daisy = list(img_dir.glob('daisy/*')) 
for i in range(len(daisy)):
    inPath = str(img_dir) +'\\daisy\\'+ str(daisy[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'daisy\\'+str(daisy[i]).split('\\')[-1]
    if ran<.2:
        outPath = tst_dir+outPath
    else:
        outPath = trval_dir+outPath
    shutil.copyfile(inPath, outPath)
dandelion = list(img_dir.glob('dandelion/*')) 
for i in range(len(dandelion)):
    inPath = str(img_dir) +'\\dandelion\\'+ str(dandelion[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'dandelion\\'+str(dandelion[i]).split('\\')[-1]
    if ran<.2:
        outPath = tst_dir+outPath
    else:
        outPath = trval_dir+outPath
    shutil.copyfile(inPath, outPath)
sunflowers = list(img_dir.glob('sunflowers/*')) 
for i in range(len(sunflowers)):
    inPath = str(img_dir) +'\\sunflowers\\'+ str(sunflowers[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'sunflowers\\'+str(sunflowers[i]).split('\\')[-1]
    if ran<.2:
        outPath = tst_dir+outPath
    else:
        outPath = trval_dir+outPath
    shutil.copyfile(inPath, outPath)
tulips = list(img_dir.glob('tulips/*')) 
for i in range(len(tulips)):
    inPath = str(img_dir) +'\\tulips\\'+ str(tulips[i]).split('\\')[-1]
    ran = random.random()
    outPath = 'tulips\\'+str(tulips[i]).split('\\')[-1]
    if ran<.2:
        outPath = tst_dir+outPath
    else:
        outPath = trval_dir+outPath
    shutil.copyfile(inPath, outPath)

