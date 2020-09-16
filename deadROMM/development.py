import numpy as np
import pandas as pd
import time
import sys
import os
import importlib
from deadROMM import possumPolish

importlib.reload(possumPolish)
model = possumPolish.Project()
config_path = model.load('./deadROMM/profiles-colab.yaml','dv101left', './dev/possum101_11Apr-Phil-2020-04-13-diff/config.yaml') #101L





experimenter='Phil'
substitute_video = '/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/videos/11Apr_diff.mp4'
frame_indices = np.random.randint(0,7000,40)
csv_path = '/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101_11Apr-Phil-2020-04-13-diff/xma/dlc1.csv'
wd = '/Volumes/GoogleDrive/My Drive/Development/DeepLabCut/dev/possum101_11Apr-Phil-2020-04-13-diff'
swap=True
cross=True

substitute_name = os.path.splitext(os.path.basename(substitute_video))[0]
substitute_data_relpath = os.path.join("labeled-data",substitute_name)
substitute_data_abspath = os.path.join(wd,substitute_data_relpath)
df=pd.read_csv(csv_path,sep=',',header=0,dtype='float',na_values='NaN')
names_initial = df.columns.values
parts_initial = [name.rsplit('_',2)[0] for name in names_initial]
parts_unique_initial = []
for part in parts_initial:
    if not part in parts_unique_initial:
        parts_unique_initial.append(part)
if swap:
    print("Creating cam1Y-cam2Y-swapped synthetic markers")
    swaps = []
    df_sw = pd.DataFrame()
    for part in parts_unique_initial:
        name_x1 = part+'_cam1_X'
        name_x2 = part+'_cam2_X'
        name_y1 = part+'_cam1_Y'
        name_y2 = part+'_cam2_Y'
        swap_name_x1 = 'sw_'+name_x1
        swap_name_x2 = 'sw_'+name_x2
        swap_name_y1 = 'sw_'+name_y1
        swap_name_y2 = 'sw_'+name_y2
        df_sw[swap_name_x1] = df[name_x1]
        df_sw[swap_name_y1] = df[name_y2]
        df_sw[swap_name_x2] = df[name_x2]
        df_sw[swap_name_y2] = df[name_y1]
        swaps.extend([swap_name_x1,swap_name_y1,swap_name_x2,swap_name_y2])
    df = df.join(df_sw)
    print(swaps)
if cross:
    print("Creating cam1-cam2-crossed synthetic markers")
    crosses = []
    df_cx = pd.DataFrame()
    for part in parts_unique_initial:
        name_x1 = part+'_cam1_X'
        name_x2 = part+'_cam2_X'
        name_y1 = part+'_cam1_Y'
        name_y2 = part+'_cam2_Y'
        cross_name_x = 'cx_'+part+'_cam1x2_X'
        cross_name_y = 'cx_'+part+'_cam1x2_Y'
        df_cx[cross_name_x] = df[name_x1]*df[name_x2]
        df_cx[cross_name_y] = df[name_y1]*df[name_y2]
        crosses.extend([cross_name_x,cross_name_y])
    df = df.join(df_cx)
    print(crosses)
names_final = df.columns.values
parts_final = [name.rsplit('_',1)[0] for name in names_final]
parts_unique_final = []
for part in parts_final:
    if not part in parts_unique_final:
        parts_unique_final.append(part)
print("Importing markers: ")
print(parts_unique_final)
unique_frames_set = {}
unique_frames_set = {index for index in frame_indices if index not in unique_frames_set}
unique_frames = sorted(unique_frames_set)
print("Importing frames: ")
print(unique_frames)
df['frame_index']=[os.path.join(substitute_data_relpath,'img'+str(index).zfill(4)+'.png') for index in unique_frames]
df['scorer']=experimenter
df = df.melt(id_vars=['frame_index','scorer'])
new = df['variable'].str.rsplit("_",n=1,expand=True)
df['variable'],df['coords'] = new[0], new[1]
df=df.rename(columns={'variable':'bodyparts'})
df['coords']=df['coords'].str.rstrip(" ").str.lower()
cat_type = pd.api.types.CategoricalDtype(categories=parts_unique_final,ordered=True)
df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype(cat_type)
newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
newdf.index.name=None
newdf.to_hdf('/Users/phil/Downloads/test1.h5', 'df_with_missing', format='table', mode='w')


df=pd.read_hdf('/Users/phil/Downloads/test1.h5')
bodyparts_XY = names_initial
