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

class Project:
    def __init__(self):
        self.profile_path = r'.\profiles.yaml'
        self.num_to_extract = 20
        self.corner2move2 = 512
        self.outlier_epsilon = 30
        self.outlier_algo = 'fitting'
        self.yaml = None
        self.env_path = None
        self.wd = None
        self.experiment = None
        self.experimenter = None
        self.markers = []
        self.dirs = {}
        self.vids_separate = []
        self.vids_merged = []
        self.env = {}
        self.dlc = dlc

    def load(self, profile=None, yaml=None):
        # Interactively specifies existing project config path, or starts new project.
        if profile:
            id = profile
        else:
            id = input("Select an existing profile in profiles.yaml or type 'quit', add the profile, and come back")
        self.getProfile(id)
        if yaml:
            self.yaml = yaml
            self.getEnv()
        else:
            status = input("Type 'new' to start a new project, or enter the full path to an existing config.yaml to continue with an existing project. Type 'quit' to quit.").strip('"')
            if status == "new":
                self.createExtractMatch()
            elif status == "quit":
                sys.exit("Pipeline terminated.")
            else:
                self.yaml = status
                self.getEnv()

    def getEnv(self):
        self.wd = os.path.dirname(self.yaml)
        self.getDirs()
        self.env_path = os.path.join(self.wd,'environment.yaml')
        if os.path.exists(self.env_path):
            self.env = ruamel.yaml.load(open(self.env_path))
        else:
            self.env = self.updateEnv()
        return self.env

    def getOutliers(self, make_labels=False):
        if make_labels:
            self.deleteLabeledFrames(self.dirs['labeled'])
        # Runs 2 passes of extract_outlier_frames, focusing on distal markers for cam1 and cam2 respectively.
        dlc.extract_outlier_frames( self.yaml,
                                    self.vids_merged,
                                    outlieralgorithm=self.outlier_algo,
                                    extractionalgorithm='kmeans',
                                    automatic=True,
                                    epsilon=self.outlier_epsilon,
                                    comparisonbodyparts=[self.markers[20], self.markers[22], self.markers[24], self.markers[28],self.markers[30],self.markers[34]],
                                    savelabeled=make_labels   )
        dlc.extract_outlier_frames( self.yaml,
                                    self.vids_merged,
                                    outlieralgorithm=self.outlier_algo,
                                    extractionalgorithm='kmeans',
                                    automatic=True,
                                    epsilon=self.outlier_epsilon,
                                    comparisonbodyparts=[self.markers[21], self.markers[23], self.markers[25], self.markers[29],self.markers[31],self.markers[35]],
                                    savelabeled=make_labels   )
        self.dirs['spliced'] = os.path.join(self.dirs['labeled'],os.path.splitext(os.path.basename(self.vids_merged[0]))[0])
        self.env['outlier_indices'] = self.getOutlierIndices(self.dirs['spliced'])
        self.env['outlier_frames'] = self.extractMatchedFrames(self.env['outlier_indices'], output_dir = self.dirs['xma'], src_vids = self.vids_separate, folder_suffix='_outlier')
        self.updateEnv()
        self.splitDlc2Xma(os.path.join(self.dirs['spliced'],'machinelabels-iter0.h5'), self.markers)



    def createExtractMatch(self):
        # Creates new DeepLabCut project, overwrites default config.yaml, and performs initial frame extraction.
        task = input("Name your project (no spaces, no periods)")
        wd = input("Project path?").strip('"')
        vid_format = os.path.splitext(self.vids_separate[0])[1]
        self.yaml = dlc.create_new_project(task,self.experimenter,self.vids_separate, working_directory=wd, videotype=vid_format, copy_videos=True)
        self.updateConfig(bodyparts=self.markers, numframes2pick=self.num_to_extract, corner2move2=self.corner2move2)
        dlc.extract_frames(self.yaml, userfeedback=False)  #puts extracted frames in .\labeled-data\{video-name}
        self.getEnv()
        self.env['extracted_indices'] = self.matchFrames(self.dirs['labeled']) #get indices of extracted frames
        self.env['extracted_frames'] = self.extractMatchedFrames(self.env['extracted_indices'], output_dir = self.dirs['xma'], src_vids = self.vids_separate, folder_suffix='_matched')
        self.updateEnv()

    def getDirs(self):
        # Stores paths to directories created by DeepLabCut.create_new_project. Makes new directory to store frames extracted for XMALab.
        self.dirs['model'] = os.path.join(self.wd,"dlc-models")
        self.dirs['evaluation'] = os.path.join(self.wd,"evaluation-results")
        self.dirs['labeled'] = os.path.join(self.wd,"labeled-data")
        self.dirs['training'] = os.path.join(self.wd,"training-datasets")
        self.dirs['video'] = os.path.join(self.wd,"videos")
        self.dirs['xma'] = os.path.join(self.wd,"frames-for-xmalab")
        self.dirs['spliced'] = None
        os.makedirs(self.dirs['xma'], exist_ok=True)

    def getProfile(self, id):
        # Interactively chooses an animal/experiment profile from a list of presets (profiles.yaml).
        profiles = ruamel.yaml.load(open(self.profile_path))
        if id == "quit":
            sys.exit("Pipeline terminated.")
        else:
            if id in profiles:
                spec_data = profiles[id]
                markers = spec_data['markers']
                for marker in markers:
                    self.markers.append(marker+'_cam1')
                    self.markers.append(marker+'_cam2')
                self.experiment = spec_data['experiment']
                self.experimenter = spec_data['experimenter']
                self.vids_separate = spec_data['vids_separate']
                self.vids_merged = spec_data['vids_merged']
            else:
                sys.exit("Profile does not exist in profiles.yaml")

    def updateConfig(self, videos=[],bodyparts=[],numframes2pick=None,corner2move2=None):
        # Updates config.yaml with arguments (if supplied).
        config = ruamel.yaml.load(open(self.yaml))
        if videos:
            video_sets={video:{"crop":"0, 1024, 0, 1024"} for video in videos}
            config['video_sets']=video_sets
        if bodyparts:
            config['bodyparts']=bodyparts
        if numframes2pick:
            config['numframes2pick']=numframes2pick
        if corner2move2:
            config['corner2move2']=[corner2move2,corner2move2]
        ruamel.yaml.round_trip_dump(config, sys.stdout)
        with open(self.yaml, 'w') as file:
            ruamel.yaml.round_trip_dump(config, file)

    def updateEnv(self):
        # Stores frame indices and paths in the project directory as environment.yaml.
        with open(self.env_path, 'w') as file:
            ruamel.yaml.dump(self.env, file)
        return self.env

    def scanDir(self, directory, extension, filters=[], filter_out=True, verbose=False):
        # Recurses through a directory looking for files with a certain extension, with optional filtering behavior. Returns list of paths.
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
                    file_list = [file for file in file_list if not re.search(string, os.path.basename(file))]
            else:
                for string in filters:
                    file_list = [file for file in file_list if re.search(string, os.path.basename(file))]
        return(file_list)

    def vidToPngs(self, video_path, output_dir=None, indices_to_match=[], name_from_folder=True):
        # Takes a list of frame numbers and exports matching frames from a video as pngs.
        frame_index = 0
        frame_counter = 0
        png_list = []
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
                    png_list.append(png_path)
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
        return png_list

    def matchFrames(self, extracted_dir):
        # Recurses through a directory of pngs looking for unique frame numbers, returns a list of indices.
        extracted_files = self.scanDir(extracted_dir, extension='png')
        extracted_indices = [int(os.path.splitext(os.path.basename(png))[0][3:].lstrip('0')) for png in extracted_files]
        unique_indices = list({index for index in extracted_indices})
        result = sorted(unique_indices)
        return result

    def extractMatchedFrames(self, indices, output_dir, src_vids=[], folder_suffix=None):
        # Given a list of frame indices and a list of source videos, produces one folder of matching frame pngs per source video.
        extracted_frames = []
        for video in src_vids:
            if folder_suffix:
                out_name = os.path.splitext(os.path.basename(video))[0]+folder_suffix
            else:
                out_name = os.path.splitext(os.path.basename(video))[0]
            output = os.path.join(output_dir,out_name)
            frames_from_vid = self.vidToPngs(video, output, indices_to_match=indices, name_from_folder=False)
            extracted_frames.append(frames_from_vid)
        return extracted_frames

    def getOutlierIndices(self, dir):
    	extracted_files = self.scanDir(dir, extension='png')
    	extracted_indices = [int(os.path.splitext(os.path.basename(png))[0][3:].lstrip('0').rstrip('labeled')) for png in extracted_files]
    	new_indices = [index for index in extracted_indices if not index in self.env['extracted_indices']]
    	unique_indices = list({index for index in new_indices})
    	result = sorted(unique_indices)
    	return result

    def importXma(self):
        # Interactively imports labels from XMALab by substituting frames from merged video for original raw frames. Updates config.yaml to point to substituted video.
        csv_path = input("Enter the full path to XMALab 2D XY coordinates csv, or type 'quit' to abort.").strip('"')
        if csv_path == "quit":
            sys.exit("Pipeline terminated.")
        else:
            self.dirs['spliced'], spliced_markers = self.spliceXma2Dlc(self.vids_merged[0], csv_path, self.env['extracted_indices'])
            self.extractMatchedFrames(self.env['extracted_indices'], output_dir=self.dirs['labeled'], src_vids=self.vids_merged)
            self.updateConfig(videos=self.vids_merged, bodyparts=spliced_markers)

    def importXmaOutliers(self):
        # Interactively imports digitized outlier frames from XMALab.
        csv_path = input("Enter the full path to XMALab 2D XY coordinates csv, or type 'quit' to abort.").strip('"')
        if csv_path == "quit":
            sys.exit("Pipeline terminated.")
        else:
            self.deleteLabeledFrames(self.dirs['labeled'])
            self.spliceXma2Dlc(self.vids_merged[0], csv_path, self.env['outlier_indices'], outlier_mode=True)
        print("imported digitized outliers! merging datasets...")
        dlc.merge_datasets(self.yaml)NONONO
        print("merged imported outliers. creating new training set...")

    def mergeOutliers(self):
        # Merged CollectedData_[experimenter].h5 with MachineLabelsRefine.h5. Renames original CollectedData_[experimenter].h5 and sets it aside.
        df0 = pd.read_hdf(os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".h5")),"df_with_missing")
        df1 = pd.read_hdf(os.path.join(self.dirs['spliced'],("MachineLabelsRefine.h5")),"df_with_missing")
        df_combined = pd.concat([df0, df1])
        df_combined.sort_index(inplace=True)
        os.rename(os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".h5")),os.path.join(self.dirs['spliced'],("old_CollectedData_"+self.experimenter+".h5")))
        os.rename(os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".csv")),os.path.join(self.dirs['spliced'],("old_CollectedData_"+self.experimenter+".csv")))
        df_combined.to_hdf(
            os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".h5")),
            key="df_with_missing",
            mode="w"
        )
        df_combined.to_csv(
            os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".csv"))
        )

    def deleteLabeledFrames(self, dir):
        frame_list = self.scanDir(dir, extension='png', filters=['labeled'], filter_out=False)
        for frame in frame_list:
            os.remove(frame)
        print("deleted "+str(len(frame_list))+" labeled frames!")


    def spliceXma2Dlc(self, substitute_video, csv_path, frame_indices, outlier_mode=False):
        # Takes csv of XMALab 2D XY coordinates from 2 cameras, outputs spliced hdf+csv data for DeepLabCut
        substitute_name = os.path.splitext(os.path.basename(substitute_video))[0]
        substitute_data_relpath = os.path.join("labeled-data",substitute_name)
        substitute_data_abspath = os.path.join(self.wd,substitute_data_relpath)
        df=pd.read_csv(csv_path,sep=',',header=0,dtype='float',na_values='NaN')
        names = df.columns.values
        parts = [name.rsplit('_',1)[0] for name in names]
        parts_unique = []
        for part in parts:
            if not part in parts_unique:
                parts_unique.append(part)
        df['frame_index']=[os.path.join(substitute_data_relpath,'img'+str(index).zfill(4)+'.png') for index in frame_indices]
        df['scorer']=self.experimenter
        df = df.melt(id_vars=['frame_index','scorer'])
        new = df['variable'].str.rsplit("_",n=1,expand=True)
        df['variable'],df['coords'] = new[0], new[1]
        df=df.rename(columns={'variable':'bodyparts'})
        df['coords']=df['coords'].str.rstrip(" ").str.lower()
        cat_type = pd.api.types.CategoricalDtype(categories=parts_unique,ordered=True)
        df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype(cat_type)
        newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
        newdf.index.name=None
        if not os.path.exists(substitute_data_abspath):
            os.mkdir(substitute_data_abspath)
        if outlier_mode:
            data_name = os.path.join(substitute_data_abspath,("MachineLabelsRefine.h5"))
        else:
            data_name = os.path.join(substitute_data_abspath,("CollectedData_"+self.experimenter+".h5"))
        newdf.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
        newdf.to_csv(data_name.split('.h5')[0]+'.csv')
        print("saved "+str(data_name))
        return substitute_data_abspath, parts_unique

    def splitDlc2Xma(self, hdf_path, bodyparts):
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
