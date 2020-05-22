import os
import re
import sys
import warnings
import importlib
import cv2
import ruamel.yaml
import numpy as np
import pandas as pd
import deeplabcut as dlc
import projectSpecs

def getEnv():
    markers, experiment_id, experimenter, vids_separate, vids_merged = getSpecs(r'.\projectSpecs.yaml')
    status = input("Type 'new' to start a new project, or enter the path to an existing config.yaml to continue with an existing project.").strip('"')
    if status == "new":
        createExtractMatch(20)
    else:
        path_config_file = status
    project_path = os.path.dirname(path_config_file)
    extracted_dir = os.path.join(project_path,"labeled-data")
    xma_frames_dir = os.path.join(project_path,"frames-for-xmalab")
    config = {  'yaml': path_config_file,
                'wd': project_path,
                'extracted_dir': extracted_dir,
                'xma_dir': xma_frames_dir,
                'markers': markers,
                'experiment_id': experiment_id,
                'experimenter': experimenter,
                'vids_separate': vids_separate,
                'vids_merged': vids_merged   }
    return config

def createExtractMatch(num_to_extract=20):
    task = input("Name your project (no spaces, no periods)")
    wd = input("Project path?").strip('"')
    vid_format = os.path.splitext(vids_separate[0])[1]
    path_config_file = dlc.create_new_project(task,experimenter,vids_separate, working_directory=wd, videotype=vid_format, copy_videos=True)
    path_config_file = updateConfig(path_config_file, numframes2pick=num_to_extract)
    dlc.extract_frames(path_config_file, userfeedback=False)  #puts extracted frames in .\labeled-data\{video-name}
    dirs = getDirs(path_config_file)
    extracted_indices = matchFrames(dirs.labeled) #get indices of extracted frames
    extracted_frames = extractMatchedFrames(extracted_indices, output_dir = dirs.xma, src_vids=

def getDirs(path_config_file):
    project_path = os.path.dirname(path_config_file)
    model = os.path.join(project_path,"dlc-models")
    evaluation = os.path.join(project_path,"evaluation-results")
    labeled = os.path.join(project_path,"labeled-data")
    training = os.path.join(project_path,"training-datasets")
    video = os.path.join(project_path,"videos")
    xma_frames = os.path.join(project_path,"frames-for-xmalab")
    os.makedirs(xma_frames_dir, exist_ok=True)
    dirs = {    'wd':project_path,
                'model':model,
                'evaluation':evaluation,
                'labeled':labeled,
                'training':training,
                'video':video,
                'xma':xma_frames}
    return dirs

# def getEnv():
#     status = input("Type 'new' to start a new project, otherwise hit ENTER.")
#     if status == "new":
#         task = input("Name your project (no spaces, no periods)")
#         experimenter = input("Name of experimenter?")
#         wd = input("Project path?").strip('"')
#         vid_list = []
#         first_vid = input("Path to first video?").strip('"')
#         vid_list.append(first_vid)
#         next_vid = input("Path to next video? Leave blank and hit ENTER if done.").strip('"')
#         while next_vid:
#             vid_list.append(next_vid)
#             next_vid = input("Path to next video? Leave blank and hit ENTER if done.").strip('"')
#             if next_vid=="":
#                 break
#         vid_format = os.path.splitext(vid_list[0])[1]
#         path_config_file = dlc.create_new_project(task,experimenter,vid_list, working_directory=wd, videotype=vid_format, copy_videos=True)
#     else:
#         path_config_file = input("Path to config.yaml?").strip('"')
#     markers, vids_separate, vids_merged = getAnimalSpec()
#     project_path = os.path.dirname(path_config_file)
#     extracted_dir = os.path.join(project_path,"labeled-data")
#     xma_frames_dir = os.path.joint(project_path,"frames-for-xmalab")
#     return path_config_file, project_path, extracted_dir, xma_frames_dir

def getSpecs(path_to_specfile):
    id = input("Select an existing profile in the projectSpecs file or type 'quit', add the profile, and come back")
    if id == "quit":
        return
    else:
        spec_data = ruamel.yaml.load(open(path_to_specfile))[id]
        markers = spec_data['markers']
        double_markers = []
        for marker in markers:
            double_markers.append(marker+'_cam1')
            double_markers.append(marker+'_cam2')
        experiment_id = spec_data['experiment_id']
        experimenter = spec_data['experimenter']
        vids_separate = spec_data['vids_separate']
        vids_merged = spec_data['vids_merged']
    return double_markers, experiment_id, experimenter, vids_separate, vids_merged

#
#
# def getSpecs():
#     id = input("Select an existing profile in the specs file or type 'quit', add the profile, and come back")
#     if id == "quit":
#         return
#     else:
#         markers = getattr(projectSpecs, id)['markers']
#         double_markers = []
#         for marker in markers:
#             double_markers.append(marker+'_cam1')
#             double_markers.append(marker+'_cam2')
#     return double_markers


def updateConfig(path_config_file, videos=[],bodyparts=[], numframes2pick=20, corner2move2=512):
    config = ruamel.yaml.load(open(path_config_file))
    if videos:
        video_sets={video:{"crop":"0, 1024, 0, 1024"} for video in videos}
        config['video_sets']=video_sets
    if bodyparts:
        config['bodyparts']=bodyparts
    config['numframes2pick']=numframes2pick
    config['corner2move2']=[corner2move2,corner2move2]
    ruamel.yaml.round_trip_dump(config, sys.stdout)
    with open(path_config_file, 'w') as fp:
        ruamel.yaml.round_trip_dump(config, fp)
        fp.close()
    return path_config_file


def scanDir(directory, extension='avi', filters=[], filter_out=True, verbose=False):
    file_list=[]
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(extension):
                filename = os.path.join(root, name)
                if verbose == True:
                    print("Found file with extension ."+ extension + ": " + filename)
                file_list.append(filename)
                continue
            else:
                continue
    if len(filters) != 0:
        if filter_out==True:
            for string in filters:
                file_list = [file for file in file_list if not re.search(string, file)]
        else:
            for string in filters:
                file_list = [file for file in file_list if re.search(string, file)]
    return(file_list)


def vidToPngs(video_path, output_dir=None, indices_to_match=[], name_from_folder=True):
    frame_index = 0
    frame_counter = 0
    if name_from_folder:
        out_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        out_name = 'img'
    if output_dir==None:
        out_dir = os.path.join(os.path.dirname(video_path),out_name)
    else:
        out_dir = output_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = round(cap.get(5),2)
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_counter += 1
        if ret == True:
            if indices_to_match and not frame_index in indices_to_match:
                frame_index += 1
                continue
            else:
                png_name = out_name+str(frame_index).zfill(4)+'.png'
                png_path = os.path.join(out_dir, png_name)
                cv2.imshow('frame',frame)
                cv2.imwrite(png_path, frame)
                frame_counter = 0
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_index += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("done!")


def matchFrames(extracted_dir):
    extracted_files = scanDir(extracted_dir, extension='png')
    extracted_indices = [int(os.path.splitext(os.path.basename(png))[0][3:].lstrip('0')) for png in extracted_files]
    unique_indices = []
    unique_indices = [index for index in extracted_indices if not index in unique_indices]
    result = sorted(unique_indices)
    return result

def extractMatchedFrames(indices, output_dir = None, src_vids=[]):
    extracted_frames = []
    for video in src_vids:
        out_name = os.path.splitext(os.path.basename(video))[0]+'_matched'
        if output_dir is not None:
            output = output_dir
        else:
            output = os.path.join(os.path.dirname(video),out_name)
        vidToPngs(video, output, indices_to_match=indices, name_from_folder=False)
        extracted_frames.append(output)
    return extracted_frames

def spliceXma2Dlc(substitute_video,project_path,csv_path, frame_indices):
    substitute_name = os.path.splitext(os.path.basename(substitute_video))[0]
    substitute_data_relpath = os.path.join("labeled-data",substitute_name)
    substitute_data_abspath = os.path.join(project_path,substitute_data_relpath)

    df=pd.read_csv(csv_path,sep=',',header=0,dtype='float',na_values='NaN')
    names = df.columns.values
    parts = [name.rsplit('_',1)[0] for name in names]
    parts_unique = []
    for part in parts:
        if not part in parts_unique:
            parts_unique.append(part)
    df['frame_index']=[os.path.join(substitute_data_relpath,'img'+str(index).zfill(4)+'.png') for index in frame_indices]
    df['scorer']=experimenter
    df = df.melt(id_vars=['frame_index','scorer'])
    new = df['variable'].str.rsplit("_",n=1,expand=True)
    df['variable'],df['coords'] = new[0], new[1]
    df=df.rename(columns={'variable':'bodyparts'})
    df['coords']=df['coords'].str.rstrip(" ").str.lower()
    cat_type = pd.api.types.CategoricalDtype(categories=parts_unique,ordered=True)
    df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype(cat_type)
    newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
    newdf.index.name=None
    ##export
    if not os.path.exists(substitute_data_abspath):
        os.mkdir(substitute_data_abspath)
    data_name = os.path.join(substitute_data_abspath,("CollectedData_"+experimenter+".h5"))
    newdf.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
    newdf.to_csv(data_name.split('.h5')[0]+'.csv')
    print("saved "+str(data_name))
    return substitute_data_abspath, parts_unique

def splitDlc2Xma(hdf_path, bodyparts):
    bodyparts_XY = []
    for part in bodyparts:
        bodyparts_XY.append(part+'_X')
        bodyparts_XY.append(part+'_Y')
    df=pd.read_hdf(hdf_path)
    # extracted_frames= [index for index in df.index]
    df = df.reset_index().melt(id_vars=['index'])
    df = df[df['coords'] != 'likelihood']
    df['id'] = df['bodyparts']+'_'+df['coords'].str.upper()
    df[['index','value','id']]
    df = df.pivot(index='index',columns='id',values='value')
    extracted_frames = [index.split('\\')[-1] for index in df.index]
    df = df.reindex(columns=bodyparts_XY)
    df.to_csv(os.path.splitext(hdf_path)[0]+'_split'+'.csv',index=False)
    return extracted_frames
