import pandas as pd
import scipy as sp
from scipy.signal.signaltools import correlate2d as c2d
import imageio
import os
import cv2
from skimage.metrics import structural_similarity


def calculate_meteor_counts_per_hour(file_in):
    df = pd.read_csv(file_in, delimiter='\t')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df2 = df.groupby(df['timestamp'].dt.floor('h'))['image'].count()
    return df2


def get_image(i):
    data = imageio.imread(path + i)
    # convert to grey-scale using W3C luminance calc
    data = sp.inner(data, [299, 587, 114]) / 1000.0
    # normalize per http: // en.wikipedia.org / wiki / Cross - correlation
    return (data - data.mean()) / data.std()


def get_diff():
    im1 = cv2.imread('02.jpg')
    im2 = cv2.imread('01.jpg')
    # Convert images to grayscale
    #before_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    #after_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    #(score, diff) = structural_similarity(before_gray, after_gray, full=True)
    #print("Image Similarity: {:.4f}%".format(score * 100))
    diff = 255 - cv2.absdiff(im1, im2)

    cv2.imshow('diff', diff)
    cv2.waitKey()


def compare_image_similarity(list_of_images):

    print (list_of_images)
    im1 = get_image(list_of_images[0])
    im2 = get_image(list_of_images[1])
    im3 = get_image(list_of_images[2])

    c11 = c2d(im1, im1, mode='same')  # baseline
    c12 = c2d(im1, im2, mode='same')
    c13 = c2d(im1, im3, mode='same')
    c23 = c2d(im2, im3, mode='same')
    print (c11.max(), c12.max(), c13.max(), c23.max())

path = 'C:/Users/stefanie/PycharmProjects/SuperMeteor/supermeteor/output/test/'
l = os.listdir(path)
compare_image_similarity(l)

#get_diff()