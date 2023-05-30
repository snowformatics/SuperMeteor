import glob
import time
import math as m
import random
import datetime
#import numpy as np
import pixellib
import tensorflow as tf
#from PIL import Image


print("################### Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#file_name = 'test11.jpg'
#picture = Image.open(file_name)
#picture.save("Compressed50_" + file_name, optimize=True, quality=50)

from pixellib.instance import custom_segmentation
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 3, class_names= ["BG", "Artifical-Star", "Background", "Meteor"])
segment_image.load_model("ok-mask_rcnn_model.068-0.405656.h5")

segmask, output = segment_image.segmentImage("50_percent_imagesize.jpg", show_bboxes=True, output_image_name="test_out.jpg")

#print (segmask, output)