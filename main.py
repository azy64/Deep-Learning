
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:53:18 2019

@author: azaria
"""

from SegNet import SegNet
from drawings_object import display_color_legend
#Training-----------------------------------------
segnet=SegNet()#
segnet.train(max_steps=18)
segnet.save()
segnet.visual_results(dataset_type = "TRAIN", images_index = 2, FLAG_MAX_VOTE = False)

#segnet.train()
#segnet.save()
#segnet.visual_results(dataset_type = "TRAIN", images_index = 2, FLAG_MAX_VOTE = False)
#display_color_legend()

#validation du dataset
#SegNet().visual_results("VAL", [0,50], False)
#display_color_legend()

####test data

#SegNet().visual_results("TEST", [62,66,198], False)
#display_color_legend()
