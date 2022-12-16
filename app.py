import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans

from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000
import cv2 as cv

import time
import math
import os

def select_roi(img_path):
    print('\nSelect an area of the image that best shows the color of your thread and press ENTER.')
    print('When you are happy with your selection, press any key to save it.')
    img = cv.imread(img_path)

    roi = cv.selectROI(img)

    roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    cv.imshow("ROI", roi_cropped)

    cv.imwrite("./captures/crop_img.jpg",roi_cropped)

    c = cv.waitKey(0)

    cv.destroyAllWindows()

def take_screenshot(camera_number):
    print('\nPress SPACE to take a screenshot.')
    cam = cv.VideoCapture(camera_number)

    cv.namedWindow("app")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv.imshow("app", frame)

        if cv.waitKey(1) == ord(' '):
            cv.imwrite('./captures/video_image.jpg', frame)
            break

        elif cv.waitKey(1) == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()
    
    
def start_app():
    os.system('cls')
    print('Hello! This app will help you to identify the color code of your embroidery floss using the DMC catalogue.')
    print('Upload a file or take a picture.\n')
    
def file_or_cam():
    print('>>>Press 0 to use an image file\n>>>Press 1 to use your camera web')
    file_cam = input()
    if int(file_cam)==0:
        use_file()
    elif int(file_cam)==1:
        use_cam()
    else:
        print('That is not a valid option')
        file_or_cam()

def use_cam():
    cam_num=input('>>>Press 0 to use your webcam\n>>>Press 1 to use your phone camera\n')
    if int(cam_num)==0:
        take_screenshot(1)
    elif int(cam_num)==1:
        take_screenshot(2)
    else:
        print('That is not a valid option')
        use_cam()
    select_roi('captures/video_image.jpg')
            
def use_file():
    img_path = input('Please input your image file path (ie:test_images/test_thread_1.jpg): ')
    select_roi(img_path) 
    
def rgb_to_hex(colors):
    hexcodes = []
    for i in range(3):
        x = '#%02x%02x%02x' % (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        hexcodes.append(x)
    return hexcodes

def model():
    img=cv.imread('captures/crop_img.jpg',1)
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img=img.reshape((img.shape[1]*img.shape[0],3))
    kmeans=KMeans(n_clusters=3)
    s=kmeans.fit(img)
    labels=list(kmeans.labels_)
    centroid=kmeans.cluster_centers_
    percent=[]
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
    colors = centroid.astype(int)    
    return colors
    
def color_distance(col1,col2):
    distance = math.sqrt(sum((col1[i]-col2[i])**2 for i in range(3)))
    return distance

def get_database():
    df = pd.read_csv('data/borrador_base.csv')
    return df

def get_rgb_database():
    rgb_values = []
    for i in range(df['hexcode1'].shape[0]):
        rgb_row = (df.r1[i],df.g1[i], df.b1[i])
        rgb_values.append(rgb_row)
    return rgb_values

def get_all_distances():
    all_distances = []
    for j in range(len(colors)):
        distances = []
        for i in range(len(rgb_values)):
            distance = color_distance(colors[j],rgb_values[i])
            distances.append(distance)
        all_distances.append(distances)
    return all_distances

def get_closest_color():
    found_color = []
    for i in range(len(colors)):
        min_dist = min(all_distances[i])
        mindist_index=all_distances[i].index(min_dist)
        mistery_color = df.code[mindist_index]
        found_color.append(mistery_color)
    data = Counter(found_color)
    answer = data.most_common(1)
    return answer

start_app()
time.sleep(1)
file_or_cam()
colors = model()
hexcodes = rgb_to_hex(colors)
df = get_database()
rgb_values = get_rgb_database()
all_distances = get_all_distances()
answer = get_closest_color()
time.sleep(1)
print('\n')
print('*******************************************')
print(f'Your thread most closely resembles DMC {answer[0][0]}')
print('*******************************************')
print('\n')