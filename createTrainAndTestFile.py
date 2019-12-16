#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:34:02 2019

@author: azaria
"""
from PIL import Image
import numpy as np
#from numpy import tensordot 
import matplotlib.pyplot as plt
from scipy import signal
import os
#import tensorly as tl
from functions import *
import pickle
#from tensorly.decomposition import tucker, parafac
import cv2
##dossir dans lesquels on va enreistrer nos fichier de mask en un seul channel#########
testlabel="TEST_LABELED/"
trainlabel="TRAIN_LABELED/"
validationlabel="VALIDATION_LABELED/"
if not os.path.exists(testlabel):
    os.makedirs(testlabel,exist_ok=True)
if not os.path.exists(trainlabel):
    os.makedirs(trainlabel,exist_ok=True)  
if not os.path.exists(validationlabel):
    os.makedirs(validationlabel,exist_ok=True)  
#########################################################################################
train="train.txt"
test="test.txt"
val="val.txt"
doc="stage_master/RedEdge/RedEdge/"
ex=open(train,"a")
liste_t=os.listdir("./TRAIN")
for fichier in liste_t:
    if not os.path.isdir(fichier):
        r,f=fichier.split("_")
        name,ext=f.split(".")
        name=name+"_GroundTruth_color.png"
        print(name," ",ext)
        rep="/home/azaria/Documents/"+doc+r+"/groundtruth/"+r+"_"+name
        im_tr=np.array(Image.open(rep))
        shape=im_tr.shape
        l=1
        for tmp in shape:
            l*=tmp
        #print(im_tr.shape)
        #plt.imshow(im_tr)
        #plt.show()
        if l==518400:
            im_tr = cv2.cvtColor(im_tr, cv2.COLOR_BGR2GRAY)
        chemin=trainlabel+r+"_"+name
        cv2.imwrite(trainlabel+r+"_"+name, im_tr)
        absolu=os.path.abspath("./TRAIN/"+fichier)
        absolu1=os.path.abspath(chemin)
        
        print(absolu)
        print(absolu1)
        ex.write(absolu+" "+absolu1+"\n")
        
ex.close()        
#############################################################VALIDATION############################################################
ex=open(val,"a")
liste_t=os.listdir("./VALIDATION")
for fichier in liste_t:
    if not os.path.isdir(fichier):
        r,f=fichier.split("_")
        name,ext=f.split(".")
        name=name+"_GroundTruth_color.png"
        rep="/home/azaria/Documents/"+doc+r+"/groundtruth/"+r+"_"+name
        im_tr=np.array(Image.open(rep))
        shape=im_tr.shape
        l=1
        for tmp in shape:
            l*=tmp
        if l==518400:
            im_tr = cv2.cvtColor(im_tr, cv2.COLOR_BGR2GRAY)
        chemin=validationlabel+r+"_"+name
        cv2.imwrite(validationlabel+r+"_"+name, im_tr)
        absolu=os.path.abspath("./VALIDATION/"+fichier)
        absolu1=os.path.abspath(chemin)
        
        print(absolu)
        print(absolu1)
        ex.write(absolu+" "+absolu1+"\n")
        
ex.close()     

##################################################TEST################################
ex=open(test,"a")
liste_t=os.listdir("./TEST")
for fichier in liste_t:
    if not os.path.isdir(fichier):
        r,f=fichier.split("_")
        name,ext=f.split(".")
        name=name+"_GroundTruth_color.png"
        rep="/home/azaria/Documents/"+doc+r+"/groundtruth/"+r+"_"+name
        im_tr=np.array(Image.open(rep))
        shape=im_tr.shape
        l=1
        for tmp in shape:
            l*=tmp
        if l==518400:
            im_tr = cv2.cvtColor(im_tr, cv2.COLOR_BGR2GRAY)
        chemin=testlabel+r+"_"+name
        cv2.imwrite(testlabel+r+"_"+name, im_tr)
        absolu=os.path.abspath("./TEST/"+fichier)
        absolu1=os.path.abspath(chemin)
        
        print(absolu)
        print(absolu1)
        ex.write(absolu+" "+absolu1+"\n")
        
ex.close()     