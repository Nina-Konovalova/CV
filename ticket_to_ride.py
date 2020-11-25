from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}
#img = cv2.imread("C:/Users/user/Documents/Intro 2CV/HW1/all.jpg")
img = cv2.imread(path_to_img)

def predict_image(img):
    # raise NotImplementedError
    city_centers = [[1000, 2000], [1500, 3000], [1204, 3251]]
    n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
    scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    #------------------------------------
   
    img = cv2.GaussianBlur(img,(7,5),0)
    img = img[..., ::-1]
    #------------------------------------
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]
    #red mask
    mask_red = ( (HUE > 160)  ) & (SAT > 120) &  (SAT < 210)# & (LIGHT > 150)
    mask_int_red = mask_red.astype(np.uint8)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((15,15))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_OPEN, kernel)
    #Red conturs
    mask_int_red = cv2.medianBlur(mask_int_red,5)
    contours, hierarchy = cv2.findContours(mask_int_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_trains['red'] = int(np.sum([cv2.contourArea(cnt) for  cnt in contours])/3550)
    #-----------------------------------------------
    #green trains
    mask_green = ( (HUE > 30) & (HUE < 90) ) & (LIGHT < 69) & (LIGHT > 40) & ((SAT > 100) & (SAT < 230))
    mask_int_green = mask_green.astype(np.uint8)
    kernel = np.ones((7,7))
    mask_int_green = cv2.morphologyEx(mask_int_green, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((7,7))
    mask_int_green = cv2.morphologyEx(mask_int_green, cv2.MORPH_OPEN, kernel)
    mask_int_green = cv2.medianBlur(mask_int_green,5)
    contours, hierarchy = cv2.findContours(mask_int_green, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n_trains['green'] = int(np.sum([cv2.contourArea(cnt) for  cnt in contours])/3550)
    #------------------------------------------------------------------
    #yellow trains
    mask_yellow = ( (HUE > 15) & (HUE < 30) )  & (SAT > 144)#& (LIGHT > 83)# & (LIGHT < 70)# & ((SAT > 150) & (SAT < 250)) & (LIGHT < 83)
    mask_int_yellow = mask_yellow.astype(np.uint8)
    kernel = np.ones((1,1))*60
    mask_int_yellow= cv2.morphologyEx(mask_int_yellow, cv2.MORPH_OPEN, kernel)
    mask_int_yellow = cv2.medianBlur(mask_int_yellow,5)
    contours, hierarchy = cv2.findContours(mask_int_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_trains['yellow'] = int(np.sum([cv2.contourArea(cnt) for  cnt in contours])/3550)
    #---------------------------------------------------------------------------
    #blue trains
    mask_blue = ( (HUE > 90) & (HUE < 120) ) & ((SAT > 120) & (SAT < 250)) & (LIGHT < 80)
    mask_int_blue = mask_blue.astype(np.uint8)
    mask_int_blue = cv2.medianBlur(mask_int_blue,5)
    mask_int_blue = cv2.GaussianBlur(mask_int_blue,(7,5),0)
    contours, hierarchy = cv2.findContours(mask_int_blue, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n_trains['blue'] = int(np.sum([cv2.contourArea(cnt) for  cnt in contours])/3400)
    #-------------------------------------------------------------------------
    #black trains
    mask_black = (LIGHT < 25)
    mask_int_black = mask_black.astype(np.uint8)
    kernel = np.ones((15,15))
    mask_int_black = cv2.morphologyEx(mask_int_black, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((19,19))
    mask_int_black = cv2.morphologyEx(mask_int_black, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask_int_black, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    n_trains['black'] = int(np.sum([cv2.contourArea(cnt) for  cnt in contours])/3600)
    
    #--------------------------------------------------------------------------
    #city centers
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(7,5),0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp = 1.08,param1=60, param2=52, 
                           minDist = 100, minRadius = 12, maxRadius = 35)
    city_centers = circles[:,:,0:-1]
    return city_centers, n_trains, scores
centers, n_trains, scores = predict_image(img)