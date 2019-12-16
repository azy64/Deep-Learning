#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:27:39 2019

@author: azaria
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:24:07 2019

@author: azaria
"""

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
doc="stage_master/"
rp="TEST/"   
if not os.path.exists(rp):
    os.makedirs(rp,exist_ok=True)             
def parcour():
   num=["003"]
   target="/tile/"
   chemin="/home/azaria/Documents/"+doc+"RedEdge/RedEdge/"
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
               print(imageName)
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
           #print(u+imageName,d+imageName,t+imageName,q+imageName)
           tensor = tl.tensor(tenseur.astype(float))
           #print("la somme:",LA.norm(list_images[0],"fro"))
           #print("matrice:",list_images[0])
           if isNullMat(list_images[0])==False:
               print("passable:",u+imageName)
               #factors = parafac(tensor, rank=rang,init='random', tol=10e-6)
               #core, tucker_factors = tucker(tensor, ranks=tucker_rank, init='random', tol=10e-6)
               #print(tenseur)
               #data=(core,tucker_factors)
               #pickle.dump(factors, open("HOSVD/PARAFAC/"+numero+"_"+imageName+".pckl", 'wb'))
               #pickle.dump(data, open("HOSVD/TUCKER/"+numero+"_"+imageName+".pckl", 'wb'))
               print("voici le path:",rp+numero+"_"+imageName," et la valeur de rp:",rp)
               saver=Image.fromarray(tenseur)
               saver.save(rp+numero+"_"+imageName)
           list_images=[]
          
def isNotIn(val):
    black_list=[0,1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,25,26,27,28,29,30,31,38,39,40,41,42,43,44,52,53,54,55,56,65,66,67,68,77,78,79,80,81,90,91,92,93,101,102,103,104,105,113,114,115,116,117,118,125,126,127,128,129,130,138,139,140,141,142,150,151,152,153,154,155,163,164,165,166,167,168,175,176,177,178,179,180,181,182,187,188,189,190,191,192,193,194,195,200,201,202,203,204,205,206,207,208,209,211,212,213,214,215,216,217,218,219,220]
    if val in black_list:
        return False
    else:
        return True
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
    