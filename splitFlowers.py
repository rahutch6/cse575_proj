import os
import numpy as np 
import pathlib
import random
import shutil

img_dir = '.\\image_recog_src\\flower_photos' 
img_dir = pathlib.Path(img_dir)
tst_dir = '.\\image_recog_src\\test\\'
trval_dir = '.\\image_recog_src\\train_validate\\'
roses = list(img_dir.glob('roses/*')) 
sunflowers = list(img_dir.glob('sunflowers/*')) 
tulips = list(img_dir.glob('tulips/*')) 
print(f'og roses: {len(roses)}\n\ttest roses: {len(list(pathlib.Path(tst_dir).glob("roses/*")))}\n\ttr/val roses: {len(list(pathlib.Path(trval_dir).glob("roses/*")))}')
print(f'og sunflowers: {len(sunflowers)}\n\ttest sunflowers: {len(list(pathlib.Path(tst_dir).glob("sunflowers/*")))}\n\ttr/val sunflowers: {len(list(pathlib.Path(trval_dir).glob("sunflowers/*")))}')
print(f'og tulips: {len(tulips)}\n\ttest tulips: {len(list(pathlib.Path(tst_dir).glob("tulips/*")))}\n\ttr/val tulips: {len(list(pathlib.Path(trval_dir).glob("tulips/*")))}')
print("DO NOT RUN THIS AGAIN UNLESS TEST DIRECTORY AND TRAIN_VALIDATE DIRECTORY ARE EMPTY")
exit()
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
        # if ran<1:
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
        # if ran<1:
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
        # if ran<1:
            outPath = trval_dir+outPath
            shutil.copyfile(inPath, outPath)
print(f'og roses: {len(roses)}\n\ttest roses: {len(list(pathlib.Path(tst_dir).glob("roses/*")))}\n\ttr/val roses: {len(list(pathlib.Path(trval_dir).glob("roses/*")))}')
print(f'og sunflowers: {len(sunflowers)}\n\ttest sunflowers: {len(list(pathlib.Path(tst_dir).glob("sunflowers/*")))}\n\ttr/val sunflowers: {len(list(pathlib.Path(trval_dir).glob("sunflowers/*")))}')
print(f'og tulips: {len(tulips)}\n\ttest tulips: {len(list(pathlib.Path(tst_dir).glob("tulips/*")))}\n\ttr/val tulips: {len(list(pathlib.Path(trval_dir).glob("tulips/*")))}')
