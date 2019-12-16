#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:33:22 2019

@author: azaria
"""
from PIL import Image
import numpy as np
from numpy import tensordot 
import matplotlib.pyplot as plt
from scipy import signal
import tensorly as tl
from numpy import linalg as La
import pickle
from sklearn.decomposition import FastICA, PCA

def constitutionIn3Band(liste=[]):
    total=np.zeros((liste[0].shape[0],liste[0].shape[1],3),dtype=np.uint8)
    for d in range(3):
        for i in range(total.shape[0]):
             for j in range(total.shape[1]):
                 total[i,j,d]=liste[d][i,j]
    return total  

def constitutionIn4Band(liste=[]):
    total=np.zeros((liste[0].shape[0],liste[0].shape[1],4),dtype=np.uint8)
    for d in range(4):
        for i in range(total.shape[0]):
             for j in range(total.shape[1]):
                 total[i,j,d]=liste[d][i,j]
    return total              

def constitutionIn5Band(liste=[]):
    total=np.zeros((liste[0].shape[0],liste[0].shape[1],5),dtype=np.uint8)
    for d in range(5):
        for i in range(total.shape[0]):
             for j in range(total.shape[1]):
                 total[i,j,d]=liste[d][i,j]
    return total  

def constitutionInAllBand(liste=[],k=3):
    print("k vaut:",k)
    total=np.zeros((liste[0].shape[0],liste[0].shape[1],k),dtype=np.uint8)
    for d in range(k):
        for i in range(total.shape[0]):
             for j in range(total.shape[1]):
                 total[i,j,d]=liste[d][i,j]
    return total  

def separationRGB(tab):
    print("---",tab.shape)
    rouge=np.copy(tab)
    vert=np.copy(tab)
    blue=np.copy(tab)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            r, v, b = tab[i, j]
            rouge[i,j]=(r,0,0)
            vert[i,j]=(0,v,0)
            blue[i,j]=(0,0,b)
    return rouge,vert,blue    
    
def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

def decomposition(image):
    
    # Compute ICA
    X=image
    ica = FastICA(n_components=5)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    #S_=S_.reshape(X.shape[0],X.shape[1])
   
    print(S_)
    # We can `prove` that the ICA model applies by reverting the unmixing.
    #assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
    
    # For comparison, compute PCA
    pca = PCA(n_components=5)
    H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
    #H=H.reshape(X.shape[0],X.shape[1])
    plt.imshow(X)
    plt.show()   
    plt.imshow(S_)
    plt.show() 
    plt.imshow(H)
    plt.show() 
   
def FrobeniusEerrorFrom2Tensor(M,M1):
    k=tl.norm(M-M1,2)**2
    result=k/tl.norm(M1,2)**2
    return result
    
def separationRGBA(tab):
    rouge=np.copy(tab)
    vert=np.copy(tab)
    blue=np.copy(tab)
    autre=np.copy(tab)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            r, v, b,a = tab[i, j]
            rouge[i,j]=(r,0,0,0)
            vert[i,j]=(0,v,0,0)
            blue[i,j]=(0,0,b,0)
            autre[i,j]=(0,0,0,a)
    return rouge,vert,blue,autre  

def separationRGBAOneDim(tab):
    rouge=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    vert=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    blue=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    autre=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            r, v, b,a = tab[i, j]
            rouge[i,j]=r
            vert[i,j]=v
            blue[i,j]=b
            autre[i,j]=a
    return rouge,vert,blue,autre  
def separationRGBOneDim(tab):
    rouge=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    vert=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    blue=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    #autre=np.zeros((tab.shape[0],tab.shape[1]),np.uint8)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            r, v, b = tab[i, j]
            rouge[i,j]=r
            vert[i,j]=v
            blue[i,j]=b
            #autre[i,j]=a
    return rouge,vert,blue 
def readFileWithPickle(name_file):
    data=""
    with open(name_file, 'r') as f1:
        data=pickle.load(f1)
    return data  
def isNilMat(mat):
    z=np.zeros(mat.shape,dtype=np.uint8)
    if mat.all() == z.all():
        return True
    else:
        return False

##################################################  