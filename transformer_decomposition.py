#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:24:07 2019

@author: azaria
"""
#########decomposition###############################
from PIL import Image
import numpy as np
from numpy import tensordot 
import matplotlib.pyplot as plt
from scipy import signal
import os
import tensorly as tl
from functions import *
import pickle
from tensorly.decomposition import tucker, parafac
from numpy import linalg as LA
tucker_rank = [100, 100, 4]
rang=4
print("parcours des dossier...")
rp="TRAIN/"
if not os.path.exists(rp):
    os.makedirs(rp,exist_ok=True)
                
def parcour():
   num=["000","001","002","004"]
   target="/tile/"
   chemin="/home/azaria/Documents/stage_master/RedEdge/RedEdge/"
   couche=["NDVI","NIR","RE"] 
   list_images=[]
   for numero in num:
       path=chemin+str(numero)+target
       u=path+"NDVI/"
       d=path+"NIR/"
       t=path+"B/"
       q=path+"G/"
       c=path+"RE/"
       s=path+"R/"
       imageName="frame"
       taille=len(os.listdir(u))
       print(path,"=====================================================",taille)
       for i in range(taille-1):
           #if isNotIn(i)==True:
           #print(isNotIn(i))
           k=i
           if len(str(k))==1:
               imageName="frame000"+str(k)+".png" 
               print(imageName)
                   
           elif len(str(k))==2:
               imageName="frame00"+str(k)+".png"
               print(imageName)
           else:
               imageName="frame0"+str(k)+".png"
                   #print(imageName)
               #print(u+imageName," ",i)    
           list_images.append(np.array(Image.open(u+imageName)))
           list_images.append(np.array(Image.open(d+imageName)))
           list_images.append(np.array(Image.open(t+imageName)))
           #list_images.append(np.array(Image.open(q+imageName)))
           #list_images.append(np.array(Image.open(c+imageName)))
           #list_images.append(np.array(Image.open(s+imageName)))
               #print(list_images)
           tenseur=constitutionInAllBand(list_images)
           print("la dimension:",tenseur.shape)
           print(u+imageName,d+imageName,t+imageName,q+imageName)
           tensor = tl.tensor(tenseur.astype(float))
           #print("la somme:",LA.norm(list_images[0],"fro"))
           print("matrice:",list_images[0])
           if isNullMat(list_images[0])==False:
               print("passable:",u+imageName)
               #factors = parafac(tensor, rank=rang,init='random', tol=10e-6)
               #core, tucker_factors = tucker(tensor, ranks=tucker_rank, init='random', tol=10e-6)
               #print(tenseur)
               #data=(core,tucker_factors)
               #pickle.dump(factors, open("HOSVD/PARAFAC/"+numero+"_"+imageName+".pckl", 'wb'))
               #pickle.dump(data, open("HOSVD/TUCKER/"+numero+"_"+imageName+".pckl", 'wb'))
               saver=Image.fromarray(tenseur)
               saver.save(rp+numero+"_"+imageName)
           list_images=[]
          
def isNullMat(m):
    rep=True
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i][j]>0:
                return False
            
    return rep        
def isNulMat(mat):
    z=np.zeros(mat.shape,dtype=np.uint8)
    if mat.all() == z.all():
        return True
    else:
        return False
    
        
parcour()       
    