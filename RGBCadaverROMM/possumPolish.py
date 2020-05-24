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

class Project:
    def __init__(self):
        self.profile_path = r'.\profiles.yaml'
        self.frame_yaml = None
        self.num_to_extract = 20
        self.corner2move2 = 512
        self.yaml = None
        self.wd = None
        self.experiment = None
        self.experimenter = None
        self.markers = []
        self.dirs = {}
        self.vids_separate = []
        self.vids_merged = []
        self.extracted_frames = []
        self.extracted_indices = []
        self.outlier_frames = []
        self.outlier_indices = []

    def getEnv(self):
        # Interactively specifies existing project config path, or starts new project.
        self.getSpecs()
        status = input("Type 'new' to start a new project, or enter the full path to an existing config.yaml to continue with an existing project. Type 'quit' to quit.").strip('"')
        if status == "new":
            self.createExtractMatch()
        elif status == "quit":
            sys.exit("Pipeline terminated.")
        else:
            self.yaml = status
            self.wd = os.path.dirname(self.yaml)
            self.getDirs()
            self.frame_yaml = os.path.join(self.wd,'frame_log.yaml')
            extracted = ruamel.yaml.load(open(self.frame_yaml))
            self.extracted_frames, self.extracted_indices = extracted['extracted_frames'], extracted['extracted_indices']

    def createExtractMatch(self):
        # Creates new DeepLabCut project, overwrites default config.yaml, and performs initial frame extraction.
        task = input("Name your project (no spaces, no periods)")
        wd = input("Project path?").strip('"')
        vid_format = os.path.splitext(self.vids_separate[0])[1]
        self.yaml = dlc.create_new_project(task,self.experimenter,self.vids_separate, working_directory=wd, videotype=vid_format, copy_videos=True)
        self.wd = os.path.dirname(self.yaml)
        self.updateConfig(bodyparts=self.markers, numframes2pick=self.num_to_extract, corner2move2=self.corner2move2)
        dlc.extract_frames(self.yaml, userfeedback=False)  #puts extracted frames in .\labeled-data\{video-name}
        self.getDirs()
        self.extracted_indices = self.matchFrames(self.dirs['labeled']) #get indices of extracted frames
        self.extracted_frames = self.extractMatchedFrames(self.extracted_indices, output_dir = self.dirs['xma'], src_vids = self.vids_separate)
        self.createFrameLog()

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

    def getSpecs(self):
        # Interactively chooses an animal/experiment profile from a list of presets (profiles.yaml).
        id = input("Select an existing profile in profiles.yaml or type 'quit', add the profile, and come back")
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
        with open(self.yaml, 'w') as fp:
            ruamel.yaml.round_trip_dump(config, fp)
            fp.close()

    def createFrameLog(self):
        # Stores frame indices and paths in the project directory as frame_log.yaml.
        self.frame_yaml = os.path.join(self.wd,'frame_log.yaml')
        with open(self.frame_yaml, 'w') as fp:
            buffer = dict(extracted_indices=self.extracted_indices, extracted_frames=self.extracted_frames)
            ruamel.yaml.dump(buffer, fp)
            fp.close()

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
                    file_list = [file for file in file_list if not re.search(string, file)]
            else:
                for string in filters:
                    file_list = [file for file in file_list if re.search(string, file)]
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
        unique_indices = []
        unique_indices = [index for index in extracted_indices if not index in unique_indices]
        result = sorted(unique_indices)
        return result

    def extractMatchedFrames(self, indices, output_dir, src_vids=[], tag_folder=True):
        # Given a list of frame indices and a list of source videos, produces one folder of matching frame pngs per source video.
        extracted_frames = []
        for video in src_vids:
            if tag_folder == True:
                out_name = os.path.splitext(os.path.basename(video))[0]+'_matched'
            else:
                out_name = os.path.splitext(os.path.basename(video))[0]
            output = os.path.join(output_dir,out_name)
            frames_from_vid = self.vidToPngs(video, output, indices_to_match=indices, name_from_folder=False)
            extracted_frames.append(frames_from_vid)
        return extracted_frames

    def importXma(self):
        # Interactively imports labels from XMALab by substituting frames from merged video for original raw frames. Updates config.yaml to point to substituted frames.
        csv_path = input("Enter the full path to XMALab 2D XY coordinates csv, or type 'quit' to abort.").strip('"')
        if csv_path == "quit":
            sys.exit("Pipeline terminated.")
        else:
            self.dirs['spliced'], spliced_markers = self.spliceXma2Dlc(self.vids_merged[0], csv_path, self.extracted_indices)
            self.extractMatchedFrames(self.extracted_indices, output_dir=self.dirs['labeled'], src_vids=self.vids_merged, tag_folder=False)
            self.updateConfig(videos=self.vids_merged, bodyparts=spliced_markers)

    def spliceXma2Dlc(self, substitute_video, csv_path, frame_indices):
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
