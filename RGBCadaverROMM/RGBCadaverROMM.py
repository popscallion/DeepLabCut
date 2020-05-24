# %% Setup and parameters
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import ruamel.yaml
import sys
import os
import importlib
import tensorflow as tf
import numpy as np
import re
import possumPolish

importlib.reload(possumPolish)

model = possumPolish.Project() #0. define a new project
model.load('dv101',r"C:\Users\Phil\Development\DeepLabCut\dev\possum101_11Apr-Phil-2020-04-13-diff\config.yaml") # 1. create a new dlc project with raw videos, extract 20 frames with k-means from each video, grab 40 frames total from each of two vids, stores frame paths and indices in frame_log.yaml
#2. now go away and digitize the 40 frames in xmalab
model.importXma() #3. come back and substitute merged video for raw vids
# model.dlc.check_labels(model.yaml) ##4. optionally, check to see if labels are plotting correctly
# dlc.create_training_dataset(model.yaml, augmenter_type="imgaug") ##5. make training set
# deeplabcut.train_network(model.yaml, displayiters=10,saveiters=10000, maxiters=200000) ##6. Train network
dlc.evaluate_network(model.yaml, plotting=True)
dlc.analyze_videos(model.yaml,[r"Z:\lab\NSF forelimb project\Phil_lab\dlc-data-swap\11Apr_diff.mp4"])
dlc.create_labeled_video(model.yaml,[r"Z:\lab\NSF forelimb project\Phil_lab\dlc-data-swap\11Apr_diff.mp4"])
dlc.extract_outlier_frames(model.yaml,[r"Z:\lab\NSF forelimb project\Phil_lab\dlc-data-swap\11Apr_diff.mp4"],outlieralgorithm='jump', extractionalgorithm='kmeans', automatic=True, epsilon=30, comparisonbodyparts=['Ulna_olc_cam1','Ulna_dst_cam1','Radius_prx_cam1','Radius_dst_cam1','Humerus_ent_cam1','Humerus_ect_cam1'])



model.getOutliers()
model.env


##make create training dataset and train
###SUNDAY:
#extract outlier frames from individual vids, and bundle into a function
#digitize outliers and retrain model
#bundle training and evaluation into function
###MONDAY:
#refine
store dir paths in framelog.yaml. write for new project and update for existing
##wishlist: mover helper to overwrite project paths in pose_cfg and config


trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file,1,0.95)
cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)

cfg_dlc['scale_jitter_lo']= 0.5
cfg_dlc['scale_jitter_up']=1.5

cfg_dlc['augmentationprobability']=.5
cfg_dlc['batch_size']=1 #pick that as large as your GPU can handle it
cfg_dlc['elastic_transform']=True
cfg_dlc['rotation']=180
cfg_dlc['grayscale']=False
cfg_dlc['hist_eq']=True
cfg_dlc['fliplr']=True
cfg_dlc['covering']=True
cfg_dlc['motion_blur'] = True
cfg_dlc['optimizer'] ="adam"
cfg_dlc['dataset_type']='imgaug'
cfg_dlc['multi_step']=[[1e-4, 7500], [5*1e-5, 12000], [1e-5, 50000], [5e-6, 200000]]

deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

# deeplabcut.create_training_model_comparison(path_config_file,num_shuffles=1,net_types=['resnet_50'],augmenter_types=['default','imgaug'])


# 	print("EVALUATE")
# 	deeplabcut.evaluate_network(path_config_file, Shuffles=[shuffle],plotting=True)

# 	print("Analyze Video")

# 	videofile_path = os.path.join(os.getcwd(),'openfield-Pranav-2018-10-30','videos','m3v1mp4.mp4')

# 	deeplabcut.analyze_videos(path_config_file, [videofile_path], shuffle=shuffle)

# 	print("Create Labeled Video and plot")
# 	deeplabcut.create_labeled_video(path_config_file,[videofile_path], shuffle=shuffle)
# 	deeplabcut.plot_trajectories(path_config_file,[videofile_path], shuffle=shuffle)
# %% codecell
for shuffle in [0,1]:
	print("TRAIN NETWORK", shuffle)
	deeplabcut.train_network(path_config_file, shuffle=shuffle,saveiters=10000,displayiters=200,maxiters=5,max_snapshots_to_keep=11)
# %% markdown

deeplabcut.evaluate_network(path_config_file, plotting=True)
# %% codecell
deeplabcut.analyze_videos(path_config_file,[r"Z:\lab\NSF forelimb project\Phil_lab\dlc-data-swap\11Apr_diff.mp4"])
# %% codecell
deeplabcut.create_labeled_video(path_config_file,[r"Z:\lab\NSF forelimb project\Phil_lab\dlc-data-swap\11Apr_diff.mp4"])
# %% codecell
deeplabcut.extract_outlier_frames(path_config_file,[r"Z:\lab\NSF forelimb project\Phil_lab\dlc-data-swap\11Apr_diff.mp4"],outlieralgorithm='jump', extractionalgorithm='kmeans', automatic=True, epsilon=30, comparisonbodyparts=['Ulna_olc_cam1','Ulna_dst_cam1','Radius_prx_cam1','Radius_dst_cam1','Humerus_ent_cam1','Humerus_ect_cam1'])
# %% codecell
def xmalab2dlc(run,csv_path,labeled_data_path, width=1024, h_flip=False):
    ## import XMAlab 2D exports
    df = pd.read_csv(csv_path, sep=',', header=0, dtype='float', na_values=' NaN ')
    ## coerce data into DeepLabCut hierarchical format
    df['frame_index']=df.index
    df['scorer']=experimenter
    df = df.melt(id_vars=['frame_index','scorer'])
    new = df['variable'].str.rsplit("_",n=2,expand=True)
    df['variable'],df['cam'],df['coords'] = new[0], new[1], new[2]
    df=df.rename(columns={'variable':'bodyparts'})
    df['coords']=df['coords'].str.rstrip(" ").str.lower()
    if h_flip == True:
        df['value'][df['coords']=='x']= df['value'][df['coords']=='x'].apply(lambda x:width-1-x)
    df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype("category")
    df['bodyparts'].cat.set_categories(markerlist,inplace=True)
    df['frame_index'] = ['labeled-data\\' + run+"Cam"+x[-1] + '\\img' + (f"{y:03d}") + '.png' for x, y in zip(df['cam'], df['frame_index'])]
    newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
    newdf.index.name=None
    ## go into frame folders and get frame index ##
    extracted_frames = []
    for root, dirs, files in os.walk(labeled_data_path):
        for name in files:
            if name.endswith(".png") and run in root:
                camera_id = root.split(' ')[-1][-1]
                frame_no = int(name.split('.')[0].replace('img',''))
                new_name = 'labeled-data\\'+run+"Cam"+camera_id+'\\img' + (f"{frame_no:03d}") + '.png'
                extracted_frames.append(new_name)

    ## filter by list of extracted frames
    df_extracted = newdf.filter(items=pd.Index(extracted_frames),axis=0)

    ## split new df into cams 1 and 2
    df1 = df_extracted.filter(like=run+"Cam"+"1",axis=0)
    df2 = df_extracted.filter(like=run+"Cam"+"2",axis=0)

    ## split new df into cams 1 and 2, export as h5 and csv
    for x in [1,2]:
        cam_name = run+"Cam"+str(x)
        dfx = df_extracted.filter(like=cam_name,axis=0)
        data_name = labeled_data_path+cam_name+"\\CollectedData_"+experimenter+".h5"
        dfx.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
        dfx.to_csv(data_name.split('.h5')[0]+'.csv')
        print("saved "+str(data_name))




# ## Refine Labels [optional step]
# Following the extraction of outlier frames, the user can use the following function to move the predicted labels to the correct location. Thus augmenting the training dataset.
# %% codecell
%gui wx
deeplabcut.refine_labels(path_config_file)
# %% markdown
# **NOTE:** Afterwards, if you want to look at the adjusted frames, you can load them in the main GUI by running: ``deeplabcut.label_frames(path_config_file)``
#
# (you can add a new "cell" below to add this code!)
#
# #### Once all folders are relabeled, check the labels again! If you are not happy, adjust them in the main GUI:
#
# ``deeplabcut.label_frames(path_config_file)``
#
# Check Labels:
#
# ``deeplabcut.check_labels(path_config_file)``
# %% codecell
#NOW, merge this with your original data:

deeplabcut.merge_datasets(path_config_file)
# %% markdown
# ## Create a new iteration of training dataset [optional step]
# Following the refinement of labels and appending them to the original dataset, this creates a new iteration of training dataset. This is automatically set in the config.yaml file, so let's get training!
# %% codecell
deeplabcut.create_training_dataset(path_config_file)
# %% codecell
deeplabcut.train_network(path_config_file, maxiters=200000, displayiters=100)
