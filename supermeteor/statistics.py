import pandas as pd
import scipy as sp
from scipy.signal.signaltools import correlate2d as c2d
import imageio
import os
import cv2
from skimage.metrics import structural_similarity
import numpy as np
from PIL import Image
import imagehash
import shutil

def calculate_meteor_counts_per_hour(file_in):
    df = pd.read_csv(file_in, delimiter='\t')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df2 = df.groupby(df['timestamp'].dt.floor('h'))['image'].count()
    return df2




def remove_duplicates():
    l = os.listdir(path_roi)
    c1 = 0
    c2 = 1
    duplicate_lst = []

    for i in l:
        try:
            image1_path = path_roi + l[c1]
            image2_path = path_roi + l[c2]
            #print (l[c1], l[c2])
            t1 =  l[c1].split('.')[0]
            t2 = l[c2].split('.')[0]
            if t1 == t2:
                duplicate_lst.append(l[c1].split('_roi.')[0])
            else:
                #hash = imagehash.average_hash(Image.open(image1_path))
                #hash2 = imagehash.average_hash(Image.open(image2_path))

                hash = imagehash.dhash(Image.open(image1_path))
                hash2 = imagehash.dhash(Image.open(image2_path))
                if hash - hash2 < 17:
                    #print(l[c1], l[c2], hash - hash2)
                    #os.makedirs("C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/w/" + str(c1) + '/')
                    #cv2.imwrite("C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/w/" + str(c1) + '/' + l[c1], cv2.imread(image1_path))
                    #cv2.imwrite("C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/w/" + str(c1) + '/' + l[c2], cv2.imread(image2_path))
                    duplicate_lst.append(l[c1].split('_roi.')[0])
            c1 += 1
            c2 += 1
        except IndexError:
            pass
    return duplicate_lst


def prepare_output(image_lst, duplicate_lst, out_file):
    out_file = open(out_file, 'w')
    l = []
    for image in image_lst:
        if image.endswith('org.jpg'):
            l.append(image.split('_org.jpg')[0])

    for d in duplicate_lst:
        if d in l:
            l.remove(d)

    for f in l:
        # open roi
        img = cv2.imread(path_roi + f + '_roi.jpg')
        height = img.shape[0]
        width = img.shape[1]

        # Open org
        shutil.copyfile(path_pos + f + '_org.jpg', path_capt + f + '_org.jpg')

        timestamp = image.split('_')[1]
        timestamp2 = '20' + timestamp[0:2] + '-' + timestamp[2:4] + '-' + timestamp[4:6] + ' ' + timestamp[6:12]
        out_file.write(f + '\t' + '20' + timestamp[0:2] + '\t' + timestamp[2:4] + '\t' + timestamp[4:6] + '\t' + timestamp[6:12]
                       + '\t' + timestamp2 + '\t' + str(height) + '\t' + str(width) + '\n')

path_pos = "C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/pos7/"
path_roi = 'C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/rois7/'
path_capt = "C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/7/"
duplicate_lst = remove_duplicates()
image_lst = os.listdir(path_pos)
prepare_output(image_lst, duplicate_lst, 'out7.csv')
