#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:02:26 2019

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
from tensorly.decomposition import tucker, parafac, non_negative_tucker,non_negative_parafac
from numpy import linalg as LA
import cv2
tucker_rank = [100, 100, 4]
rp="TRAIN"
rang=4
print("parcours des dossier...")
couche=["NDVI/","NIR/","RE/","R/","G/","B/"] 
num="003"
target="/tile/"
ch=os.path.abspath(".")
par="PARAFAC"
tu="TUCKER"
tmp=ch.split("/")
doc=tmp[len(tmp)-1]+"/"
chemin="/home/azaria/Documents/"+doc+"RedEdge/RedEdge/"

for i in range(len(couche)+1):
    if i>2:
        rep=rp#+str(i)
        if not os.path.exists(rep):
            os.makedirs(rep,exist_ok=True)
        #if not os.path.exists(rep+"/"+par):
          #  os.mkdir(rep+"/"+par)
       # if not os.path.exists(rep+"/"+tu):
            #os.mkdir(rep+"/"+tu)
        tmp=[couche[k] for k in range(i)]
        imageName="frame"
        list_images=[]
        list_b=[]
        path=chemin+str(num)+target
        #print(path)
        #print(tmp)
        for b in range(len(tmp)):
            list_b.append(chemin+num+target+tmp[b])
        #print(list_b)
        liste_doc=os.listdir(list_b[0])
        for fichier in liste_doc:
            if not os.path.isdir(fichier):
                for e in range(len(list_b)):
                    mat=np.array(Image.open(list_b[e]+fichier))
                    #print("-------------------------------------",mat.shape)
                    list_images.append(mat)
                print("il ya",len(list_images)," dans la liste d'images")    
                if isNilMat(list_images[0])==False:
                    tenseur=constitutionInAllBand(list_images,i)
                    print("=======on vient de former et traiter l'image ",fichier,", un tenser de ",i)
                    tensor = tl.tensor(tenseur.astype(float))
                    factors = parafac(tensor, rank=i,init='random', tol=10e-6)
                    print("la taille du facteur Parafac est:",len(factors),"pour",tmp," \net la dimension du tenseur est:",tenseur.shape,"--",tensor.shape)
                    core, tucker_factors = tucker(tensor, ranks=tucker_rank, init='random', tol=10e-6)
                    if i ==3 :
                        #plt.imsave(rep+"/"+fichier,tenseur)
                        saver=Image.fromarray(tenseur)
                        saver.save(rep+"/"+fichier)
                        #cv2.imwrite(rep+"/"+fichier, tenseur)
                    #else:
                    #    with open(rep+"/"+fichier+".pckl","wb") as f:
                    #        pickle.dump(tenseur,f)
                    #with open(rep+"/"+par+"/"+fichier+".pckl","wb") as f:
                    #    pickle.dump([tensor,factors],f)
                    """with open(rep+"/"+tu+"/"+fichier+".pckl","wb") as f:
                        pickle.dump((core,tucker_factors),f)  """  
                list_images=[]
                
    