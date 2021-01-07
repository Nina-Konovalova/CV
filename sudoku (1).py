import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
import cv2
import os
from skimage import io
from skimage.feature import match_template


SCALE = 0.33

#-----------------------------------------------------------------------------
def sudoku_solver(sud_matr):
    from sudoku1 import print_solutions

    sud_matr = list(sud_matr)
    for i in range (len(sud_matr)):
        sud_matr[i] = list(sud_matr[i])
    for i in range(len(sud_matr)):
        for j in range (9):
            if sud_matr[i][j] == -1:
                sud_matr[i][j] = None
    
    #'\n'.join([' '.join(map(str, l)) for l in sud_matr])
    print_solutions(sud_matr, max_solutions=2)



#-----------------------------------------------------------------------------



def find_mask (image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,5)
    ret3,th3 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_c = 255-th3.astype(np.uint8)
    kernel = np.ones((15,15))
    img_c = cv2.morphologyEx(img_c, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(img_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    index = []
    i = 0
    for cnt in contours:
        
        areas.append(cv2.contourArea(cnt))
        index.append(i)
        i+=1
    area_dict = dict(zip(index, areas))
    areas = np.sort(areas)
    def get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k
    mask = np.zeros((image.shape[0],image.shape[1]))
    mask = cv2.fillPoly(mask, [contours[get_key(area_dict, areas[-1])]], 255)
    return mask/255

def load_templates():

    images = [io.imread('/autograder/submission/templates/' + img,as_gray=True, plugin='matplotlib') for img in sorted(os.listdir('/autograder/submission/templates/'))]


# Filter images
    bound = 0.44
    templates = {
               #0: [cv2.resize(images [0], (50, 50)) < bound],
                1: [cv2.resize(images [1], (50, 50)) < bound, cv2.resize(images [2], (50, 50)) < bound],
                2: [cv2.resize(images [3], (50, 50)) < bound, cv2.resize(images [4], (50, 50)) < bound],
                3: [cv2.resize(images [5], (50, 50)) < bound, cv2.resize(images  [6], (50, 50)) < bound],
                4: [cv2.resize(images [7], (50, 50)) < bound, cv2.resize(images [8], (50, 50)) < bound],
                5: [cv2.resize(images [9], (50, 50)) < bound, cv2.resize(images [10], (50, 50)) < bound],
                6: [cv2.resize(images [11], (50, 50)) < bound, cv2.resize(images [12], (50, 50)) < bound],
                7: [cv2.resize(images [13], (50, 50)) < bound, cv2.resize(images [14], (50, 50)) < bound],
                8: [cv2.resize(images [15], (50, 50)) < bound, cv2.resize(images [16], (50, 50)) < bound],
                9: [cv2.resize(images [17], (50, 50)) < bound, cv2.resize(images [18], (50, 50)) < bound]                           
    }
    return templates

def finding_nums (image, templates):
    
    result = np.zeros((9,9))
    thresh = 0.25
    for i in range (9):
        for j in range (9):
            max_val, answer = 0, 0
            max_val2, answer2 = 0, 0
            #plt.imshow(get_sell(image, i, j), cmap = 'gray')
            #plt.show()
            for key in templates.keys():
                for val in range (len(templates[key])):
                    res = match_template(get_sell(image, i, j),templates[key][val], pad_input=False).max()
                    res2 = match_template(get_sell(image, i, j),templates[key][val], pad_input=True).max()
                    #print(res)
                    if res > max_val:
                        max_val = res
                        answer = key
                
                    if np.max(res2) > max_val2:
                        max_val2 = np.max(res2)
                        answer2 = key
    #                 result[i, j] = answer
                    if max_val < thresh or max_val2 < thresh:
                        result[i, j] = -1
                    else:
                        result[i, j] = answer
    #rint(result)
    return result

def normalized (image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,5)
    ret3,th3 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_c = 255-th3.astype(np.uint8)
    kernel = np.ones((15,15))
    img_c = cv2.morphologyEx(img_c, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(img_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    index = []
    i = 0
    for cnt in contours:
        
        areas.append(cv2.contourArea(cnt))
        index.append(i)
        i+=1
    area_dict = dict(zip(index, areas))
    areas = np.sort(areas)
    def get_key(d, value):
        for k, v in d.items():
            if v == value:
                return k
    rect = cv2.minAreaRect(contours[get_key(area_dict, areas[-1])])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    perimeter = cv2.arcLength(contours[get_key(area_dict, areas[-1])], True) 
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contours[get_key(area_dict, areas[-1])],epsilon,True)
    if approx.shape[0] == 4:
        box = approx.reshape(4,-1)
    rect = np.zeros((4, 2), dtype = "float32")
    s = box.sum(axis = 1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]
    diff = np.diff(box, axis = 1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_sell(binary_img, row, column, delta=25):
    w,h = binary_img.shape[0:2]
    h_s, w_s = round(h / 9), round(w / 9)
   
    #print(image[y-delta : y + delta, x - delta : x + delta].shape)
    return binary_img[w_s*row : w_s*(row+1), h_s*column : h_s*(column+1)]


def get_key(d, value):
    for k, v in d.items():
        if v == value:
            return k
def predict_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sudoku_digits = [
    ]
    mask = np.bool_(np.ones_like(image))
    # loading train image:
    #train_img_4 = cv2.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
    #rf = joblib.load('/autograder/submission/random_forest.joblib')
    mask = find_mask (image)
    warped = normalized (image)
 
    templates = load_templates()
    warped_grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    warped_grey = cv2.resize(warped_grey, (550, 550))
   
    binary_img = warped_grey < 80
    sudoku_digits.append(finding_nums(binary_img, templates))
    return mask, sudoku_digits
