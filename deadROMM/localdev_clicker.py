%cd drive/My\ Drive/Development/DeepLabCut
import deeplabcut
from deadROMM import possumPolish

import possumPolish


model = possumPolish.Project()
model.autocorrect(r'/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/dlc-models/iteration-1/possum101right_biceps_tricepsJun8-trainset95shuffle1/train/150k_DIS_1scale/16Apr_diffDLC_resnet50_possum101right_biceps_tricepsJun8shuffle1_30000_everyevery10_split__p0.01_26Oct20_00h08m39s.h5',{'cam1':'/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/frames-for-xmalab/c1_16Apr_every10','cam2':'/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/frames-for-xmalab/c2_16Apr_every10'})

model.load('./deadROMM/profiles-colab.yaml','dv101right','/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/config.yaml')

len(df1)
import pandas as pd
df1 = pd.read_hdf('/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/dlc-models/iteration-1/possum101right_biceps_tricepsJun8-trainset95shuffle1/train/150k_DIS_1scale/16Apr_diffDLC_resnet50_possum101right_biceps_tricepsJun8shuffle1_30000_everyevery10_split__p0.01_26Oct20_00h08m39s_autocorrect.h5', "df_with_missing")




model.splitDlc2Xma('/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/dlc-models/iteration-1/possum101right_biceps_tricepsJun8-trainset95shuffle1/train/150k_DIS_1scale/16Apr_diffDLC_resnet50_possum101right_biceps_tricepsJun8shuffle1_30000_everyevery10_split__p0.01_26Oct20_00h08m39s_autocorrect.h5',model.markers)




import numpy as np
import time
import keyboard
import sys
import math
import os
import importlib
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
%cd /Volumes/GoogleDrive/My Drive/Development/DeepLabCut
%matplotlib inline




model


def filter_image(self, image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.30):
    krad = krad*2+1
    image_blur = cv2.GaussianBlur(image, (krad, krad), gsigma)
    image_blend = cv2.addWeighted(image, img_wt, image_blur, blur_wt, 0)
    lut = np.array([((i/255.0)**gamma)*255.0 for i in range(256)])
    image_gamma = image_blend.copy()
    im_type = len(image_gamma.shape)
    if im_type == 2:
        image_gamma = lut[image_gamma]
    elif im_type == 3:
        image_gamma[:,:,0] = lut[image_gamma[:,:,0]]
        image_gamma[:,:,1] = lut[image_gamma[:,:,1]]
        image_gamma[:,:,2] = lut[image_gamma[:,:,2]]
    return image_gamma

def show_crop(self, src, center=search_area, scale=5, contours=None, detected_marker=None):
    if len(src.shape) < 3:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    image = src.copy().astype(np.uint8)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    if contours:
        overlay = image.copy()
        scaled_contours = [contour*scale for contour in contours]
        cv2.drawContours(overlay, scaled_contours, -1, (255,0,0),2)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    cv2.drawMarker(image, (center*scale, center*scale), color = (0,255,255), markerType = cv2.MARKER_CROSS, markerSize = 10, thickness = 1)
    if detected_marker:
        cv2.drawMarker(image, (int(detected_marker[0]*scale),int(detected_marker[1]*scale)),color = (255,0,0), markerType = cv2.MARKER_CROSS, markerSize = 10, thickness = 1)
    plt.imshow(image)
    plt.show()

indices_to_use = []
paths_as_index = False
likelihood_cutoff=0.01
search_area=15
mask_size = 5
threshold = 8
hdf_path = '/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/dlc-models/iteration-1/possum101right_biceps_tricepsJun8-trainset95shuffle1/train/150k_DIS_1scale/16Apr_diffDLC_resnet50_possum101right_biceps_tricepsJun8shuffle1_30000_everyevery10_split__p0.01_26Oct20_00h08m39s.h5'
cam_dirs = {'cam1':'/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/frames-for-xmalab/c1_16Apr_every10','cam2':'/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101right_biceps_triceps-Phil-2020-06-08/frames-for-xmalab/c2_16Apr_every10'}

def autocorrect(self, hdf_path, cam_dirs={'cam1':None,'cam2':None}, indices_to_use = [], paths_as_index = False, likelihood_cutoff=0.01, search_area=15, mask_size = 5, threshold = 8): #try 0.05 also


search_area = int(search_area + 0.5) if search_area >= 10 else 10
out_name = os.path.join(os.path.dirname(hdf_path),(os.path.splitext(os.path.basename(hdf_path))[0]+'_autocorrect.h5'))
df = pd.read_hdf(hdf_path, "df_with_missing")
if indices_to_use:
    mask_list = [os.path.join(df.index[0].rsplit('/',1)[0],index) for index in png_indices]
    df = df.loc[mask_list]
scorer = df.columns.get_level_values(0)[0]
bodyparts = df.columns.get_level_values(1)
parts_unique = []
for part in bodyparts:
    this = part.rsplit('_',1)[0]
    if not this in parts_unique:
        parts_unique.append(this)
list_of_pngs = model.scanDir(cam_dirs['cam1'], 'png')
png_names = sorted([os.path.split(png)[-1] for png in list_of_pngs])
for frame_index in [977]:
    if paths_as_index:
        im_index = os.path.split(df.iloc[frame_index].name)[-1]
    else:
        im_index = png_names[frame_index]
    for cam in ['cam1', 'cam2']:
        ##Load frame
        frame = cv2.imread(os.path.join(cam_dirs[cam],str(im_index)))
        print(os.path.join(cam_dirs[cam],str(im_index)))
        plt.imshow(frame)
        plt.show()
        frame = filter_image(frame, krad=10)

        ##Loop through all markers for each frame
        for part in parts_unique:
            ##Find point and offsets
            x_float, y_float, likelihood = df.xs(part+'_'+cam, level='bodyparts',axis=1).iloc[frame_index]
            print(part+' Camera '+cam[-1]+' Likelihood: '+str(likelihood))
            if likelihood < likelihood_cutoff:
                print('Likelihood too low; skipping')
                continue
            x_start = int(x_float-search_area+0.5)
            y_start = int(y_float-search_area+0.5)
            x_end = int(x_float+search_area+0.5)
            y_end = int(y_float+search_area+0.5)

            ##Crop image to marker vicinity
            subimage = frame[y_start:y_end, x_start:x_end]

            ##Convert To float
            subimage_float = subimage.astype(np.float32)

            ##Create Blurred image
            radius = int(1.5 * 5 + 0.5) #5 might be too high
            sigma = radius * math.sqrt(2 * math.log(255)) - 1
            subimage_blurred = cv2.GaussianBlur(subimage_float, (2 * radius + 1, radius + 1), sigma)

            ##Subtract Background
            subimage_diff = subimage_float-subimage_blurred
            subimage_diff = cv2.normalize(subimage_diff, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

            ##Median
            subimage_median = cv2.medianBlur(subimage_diff, 3)

            ##LUT
            subimage_median = filter_image(subimage_median, krad=3)

            ##Thresholding
            subimage_median = cv2.cvtColor(subimage_median, cv2.COLOR_BGR2GRAY)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(subimage_median)
            thres = 0.5 * minVal + 0.5 * np.mean(subimage_median) + threshold * 0.01 * 255
            ret, subimage_threshold =  cv2.threshold(subimage_median, thres, 255, cv2.THRESH_BINARY_INV)

            ##Gaussian blur
            subimage_gaussthresh = cv2.GaussianBlur(subimage_threshold, (3,3), 1.3)

            ##Find contours
            contours, hierarchy = cv2.findContours(subimage_gaussthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x_start,y_start))
            print("Detected "+str(len(contours))+" contours in "+str(search_area)+"*"+str(search_area)+" neighborhood of marker "+part+' in Camera '+cam[-1])
            contours_im = contours.copy()
            contours_im = [contour-[x_start, y_start] for contour in contours_im]

            ##Find closest contour
            dist = 1000
            best_index = -1
            detected_centers = {}
            for i in range(len(contours)):
                detected_center, circle_radius = cv2.minEnclosingCircle(contours[i])
                distTmp = math.sqrt((x_float - detected_center[0])**2 + (y_float - detected_center[1])**2)
                detected_centers[round(distTmp, 4)] = detected_center
                if distTmp < dist:
                    best_index = i
                    dist = distTmp
            if best_index >= 0:
                detected_center, circle_radius = cv2.minEnclosingCircle(contours[best_index])
                detected_center_im, circle_radius_im = cv2.minEnclosingCircle(contours_im[best_index])
                show_crop(subimage, contours = [contours_im[best_index]], detected_marker = detected_center_im)
                # show_crop(subimage_threshold, contours=contours_im,detected_marker = detected_center_im)
                # show_crop(subimage_gaussthresh, contours = [contours_im[best_index]], detected_marker = detected_center_im)
                df.loc[df.iloc[frame_index].name, (scorer,part+'_'+cam, ['x'])]  = detected_center[0]
                df.loc[df.iloc[frame_index].name, (scorer,part+'_'+cam, ['y'])]  = detected_center[1]
print('done! saving...')
df.to_hdf(out_name, 'df_with_missing', format='table', mode='w', nan_rep='NaN')


# play with first pass filter params
# play with thresholding
# filter contours for area then circularity
# try blobdetector
