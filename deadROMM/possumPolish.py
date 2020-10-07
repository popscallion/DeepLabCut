import os
import re
import sys
import datetime
import shutil
import warnings
import importlib
import cv2
import zipfile
from PIL import Image
from subprocess import Popen, PIPE
import blend_modes
import ruamel.yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import deeplabcut as dlc

class Project:
    def __init__(self):
        self.profile_path = None
        self.num_to_extract = 20
        self.corner2move2 = 512
        self.outlier_epsilon = 30
        self.p_bound = 0.01
        self.pcutoff = 0.5
        self.yaml = None
        self.wd = None
        self.experiment = None
        self.experimenter = None
        self.markers = []
        self.dirs = {}
        self.vids_separate = []
        self.vids_merged = []
        self.config = {}
        self.dlc = dlc
        self.in_colab = 'google.colab' in sys.modules

    def load(self, profile_path, profile, yaml=None):
        '''
        Interactively specifies existing project config path, or starts new project.
            Parameters:
                profile_path (str): Path to profiles.yaml
                profile (str): Alphanumberic individual identifier used as key in profiles.yaml, e.g. 'dv101'
                yaml (str): Path to config.yaml
            Returns:
                None
        '''
        self.profile_path = profile_path
        self.getProfile(profile)
        if yaml:
            self.yaml = yaml
            self.getDirs()
        else:
            status = input("Type 'new' to start a new project, or enter the full path to an existing config.yaml to continue with an existing project. Type 'quit' to quit.").strip('"')
            if status == "new":
                self.createExtractMatch()
            elif status == "quit":
                sys.exit("Pipeline terminated.")
            else:
                self.yaml = status
                self.getDirs()
        self.updateConfig()
        print("Successfully loaded profile "+str(profile))
        return self.yaml


    def createExtractMatch(self):
        '''
        Creates new DeepLabCut project, overwrites default config.yaml, and performs initial frame extraction.
            Parameters:
                None
            Returns:
                None
        '''
        task = input("Name your project (no spaces, no periods)")
        wd = input("Enter directory where project should be created:").strip('"')
        vid_format = os.path.splitext(self.vids_separate[0])[1]
        self.yaml = dlc.create_new_project(task,self.experimenter,self.vids_separate, working_directory=wd, videotype=vid_format, copy_videos=False)
        self.getDirs()
        self.updateConfig(bodyparts=self.markers, numframes2pick=self.num_to_extract, corner2move2=self.corner2move2, pcutoff=self.pcutoff)
        extraction_event, extracted_frames = self.updateWithFunc('extract', self.trackFiles, self.dirs['labeled'], dlc.extract_frames, self.yaml, userfeedback=False)
        extracted_frames_final = self.exciseRevise(extraction_event, self.findBlanks(extracted_frames,return_indices=True))[0]
        extracted_indices = self.matchFrames(extracted_frames_final) #get indices of extracted frames
        matched_frames = self.updateWithFunc('match_extracted', self.extractMatchedFrames, extracted_indices, output_dir = self.dirs['xma'], src_vids = self.vids_separate, folder_suffix='_matched', timestamp=True)[1]
        print("Succesfully created a new DeepLabCut project and performed initial frame extraction. Frames for XMALab are in "+str(self.dirs['xma']))

    def updateWithFunc(self, type, func, *args, **kwargs):
        '''
        Constructs a new history object from the wrapped function's return and writes it to config.yaml.
            Parameters:
                type (str): Description of the operation performed by the wrapped function (e.g. 'extract','outliers_cam1')
                func (function): Function that returns a list of file paths, (e.g. self.extractMatchedFrames)
                *args, **kwargs: Arguments passed to func
            Returns:
                list_of_files (list): List returned from func
        '''
        ts = self.getTimeStamp()
        list_of_files = func(*args, **kwargs)
        update = {ts:{'operation':type,'files':list_of_files}}
        self.updateConfig(event=update)
        print("Updated config.yaml with event "+type+" at "+ts)
        return [ts, list_of_files]

    def updateWithFiles(self, type, list_of_paths):
        '''
        Constructs a new history object from a list of file paths and writes it to config.yaml.
            Parameters:
                type (str): Description of the operation that produced list_of_paths (e.g. 'extract','outliers_cam1')
                list_of_paths (list): List of file paths
            Returns:
                None
        '''
        ts = self.getTimeStamp()
        update = {ts:{'operation':type,'files':list_of_paths}}
        self.updateConfig(event=update)
        print("Updated config.yaml with event "+type+" at "+ts)

    def getDirs(self):
        '''
        Stores paths to directories created by DeepLabCut.create_new_project as instance variables. Makes new directory to store frames extracted for XMALab.
            Parameters:
                None
            Returns:
                None
        '''
        self.wd = os.path.dirname(self.yaml)
        self.dirs['models'] = os.path.join(self.wd,"dlc-models")
        self.dirs['evaluation'] = os.path.join(self.wd,"evaluation-results")
        self.dirs['labeled'] = os.path.join(self.wd,"labeled-data")
        self.dirs['spliced'] = os.path.join(self.dirs['labeled'],os.path.splitext(os.path.basename(self.vids_merged[0]))[0])
        self.dirs['training'] = os.path.join(self.wd,"training-datasets")
        self.dirs['video'] = os.path.join(self.wd,"videos")
        self.dirs['xma'] = os.path.join(self.wd,"frames-for-xmalab")
        print("Generated absolute paths to project directories")
        os.makedirs(self.dirs['xma'], exist_ok=True)

    def trackFiles(self, directory, func, *args, **kwargs):
        '''
        Tracks files generated by wrapped function by comparing before and after snapshots of the target directory.
            Parameters:
                directory (str): Directory to watch
                func (function): Function that creates new files but does not return their paths
                *args, **kwargs: Arguments passed to func
            Returns:
                result (list): List of file paths created by func
        '''
        def getSnapshot(directory):
            snapshot = []
            for root, dirs, files in os.walk(directory):
                for name in files:
                    file_path = os.path.join(root, name)
                    snapshot.append(file_path)
            return snapshot
        def diff(pre, post):
            return set(post).difference(set(pre))
        snapshot_pre = getSnapshot(directory)
        func(*args, **kwargs)
        snapshot_post = getSnapshot(directory)
        result = list(diff(snapshot_pre, snapshot_post))
        print("Function "+str(func)+" created "+str(len(result))+" new files in directory "+str(directory))
        return result

    def getProfile(self, id):
        '''
        Interactively chooses an animal/experiment profile from a list of presets (profiles.yaml).
            Parameters:
                id (str): Alphanumberic individual identifier used as key in profiles.yaml, e.g. 'dv101'
            Returns:
                None
        '''
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
                print("Loaded profile "+id)
            else:
                sys.exit("Profile does not exist in profiles.yaml")

    def subsetPredictions(self, analysis_h5, frame_paths, condition_name):
      # frame_paths should come from match_outliers
      filenames = [os.path.basename(path) for path in frame_paths if os.path.splitext(path)[1]=='.png']
      unique_names = {}
      unique_names = {name for name in filenames if name not in unique_names }
      names_to_use = sorted(unique_names)
      paths_to_use = [os.path.join(os.path.relpath(self.dirs['spliced'],self.wd),name) for name in names_to_use]
      indices_to_use = [int(index[3:-4]) for index in names_to_use]
      df = pd.read_hdf(analysis_h5, "df_with_missing")
      filtered_df = df.iloc[indices_to_use]
      df_with_names = filtered_df.set_index(pd.Series(paths_to_use))
      ts = self.getTimeStamp()
      data_name = os.path.join(self.dirs['spliced'],"FilteredMachineLabels_"+ts)
      tracked_hdf = data_name + ".h5"
      df_with_names.to_hdf(tracked_hdf, 'df_with_missing', format='table', mode='w')
      tracked_files = [tracked_hdf]
      self.updateWithFiles('subsetPredictions_'+condition_name, tracked_files)
      print("Successfully subset model predictions; saved "+str(tracked_hdf))
      self.splitDlc2Xma(tracked_hdf, self.markers)

    def getOutliers(self, num2extract, markers2watch={'c1':[],'c2':[]}, outlier_algo='uncertain', make_labels=False):
        '''
        Independently identifies outlier frames for two cameras, extracts the corresponding frames, and converts DeepLabCut predictions for those frames to XMALab format
            Parameters:
                num2extract (int): Number of outlier frames to pick from each camera. Used to overwrite the numframes2pick paramter in config.yaml
                make_labels (bool): If true, tells DeepLabCut to create duplicate pngs with marker predictions visualized
            Returns:
                None
        '''
        def specifyOutliers(markers2watch):
          if not markers2watch['c1'] or markers2watch['c2']:
            marker_dict = {i:j for i,j in enumerate(self.config['bodyparts'])}
            print(marker_dict)
            c1markers = input("Camera 1: enter markers of interest by their numeric indices, separated by a space (e.g. 3 11 18)").rstrip(' ')
            c2markers = input("Camera 2: enter markers of interest by their numeric indices, separated by a space (e.g. 4 12 19)").rstrip(' ')
            c1list = sorted([int(ele) for ele in c1markers.split(' ')])
            c2list = sorted([int(ele) for ele in c2markers.split(' ')])
            markers2watch['c1'] = [marker_dict[index] for index in c1list]
            markers2watch['c2'] = [marker_dict[index] for index in c2list]
            print(markers2watch['c1'])
            print(markers2watch['c2'])
            confirmed = input("Are these the markers you want to track? Hit enter to continue or type 'redo' to redo.")
            if confirmed == 'redo':
              markers2watch={'c1':[],'c2':[]}
              specifyOutliers(markers2watch)
            print('Markers selected.')
            return markers2watch
        self.updateConfig(numframes2pick=num2extract)
        markers2watch = specifyOutliers(markers2watch)
        if make_labels:
            self.deleteLabeledFrames(self.dirs['labeled'])
        extraction_event_cam1, outliers_cam1 = self.updateWithFunc('outliers_cam1', self.trackFiles, self.dirs['spliced'],
                                                dlc.extract_outlier_frames,
                                                self.yaml,
                                                self.vids_merged,
                                                outlieralgorithm=outlier_algo,
                                                extractionalgorithm='kmeans',
                                                p_bound = self.p_bound,
                                                automatic=True,
                                                epsilon=self.outlier_epsilon,
                                                comparisonbodyparts = markers2watch['c1'],
                                                savelabeled=make_labels)

        extraction_event_cam2, outliers_cam2 = self.updateWithFunc('outliers_cam2', self.trackFiles, self.dirs['spliced'],
                                                dlc.extract_outlier_frames,
                                                self.yaml,
                                                self.vids_merged,
                                                outlieralgorithm=outlier_algo,
                                                extractionalgorithm='kmeans',
                                                p_bound = self.p_bound,
                                                automatic=True,
                                                epsilon=self.outlier_epsilon,
                                                comparisonbodyparts = markers2watch['c2'],
                                                savelabeled=make_labels)
        self.exciseRevise(extraction_event_cam1, self.findBlanks(outliers_cam1,return_indices=True))
        self.exciseRevise(extraction_event_cam2, self.findBlanks(outliers_cam2,return_indices=True))
        newfiles = self.config['history'][extraction_event_cam1]['files']+self.config['history'][extraction_event_cam2]['files']
        outlier_indices = self.matchFrames(newfiles)
        matched_outlier_frames = self.updateWithFunc('match_outliers', self.extractMatchedFrames, outlier_indices, output_dir = self.dirs['xma'], src_vids = self.vids_separate, folder_suffix='_outlier', timestamp=True)[1]
        h5s = [path for path in newfiles if 'h5' in path]
        print(newfiles)
        self.splitDlc2Xma(h5s[0], self.markers, newfiles)

    def updateConfig(self, project_path=None, event={}, videos=[],bodyparts=[],numframes2pick=None,corner2move2=None, pcutoff=None, increment_iteration=False):
        '''
        Writes supplied arguments to config.yaml. If called without arguments, updates config instance variable from config.yaml file without modifying the latter.
            Parameters:
                project_path (str): Corresponds to 'project_path' attribute in config.yaml
                event (dict): Corresponds to 'history' attribute in config.yaml
                videos (list): Corresponds to 'video_sets' attribute in config.yaml
                bodyparts (list): Corresponds to 'bodyparts' attribute in config.yaml
                numframes2pick (int): Corresponds to 'numframes2pick' attribute in config.yaml
                corner2move2 (int): Corresponds to 'corner2move2' attribute in config.yaml
            Returns:
                None
        '''
        self.config = ruamel.yaml.load(open(self.yaml))
        if project_path:
            self.config['project_path']=project_path
        if event:
            if 'history' not in self.config:
                self.config['history'] = {}
            self.config['history'].update(event)
        if videos:
            video_sets={video:{"crop":"0, 1024, 0, 1024"} for video in videos}
            self.config['video_sets']=video_sets
            print("Updated video_sets in config.yaml")
        if bodyparts:
            self.config['bodyparts']=bodyparts
            print("Updated bodyparts in config.yaml")
        if numframes2pick:
            self.config['numframes2pick']=numframes2pick
            print("Updated numframes2pick in config.yaml")
        if corner2move2:
            self.config['corner2move2']=[corner2move2,corner2move2]
            print("Updated corner2move2 in config.yaml")
        if pcutoff:
            self.config['pcutoff']=pcutoff
            print("Updated pcutoff in config.yaml")
        if increment_iteration:
            self.config['iteration'] += 1
            print("Incremented iteration in config.yaml")
        # ruamel.yaml.round_trip_dump(self.config, sys.stdout)
        with open(self.yaml, 'w') as file:
            ruamel.yaml.round_trip_dump(self.config, file)

    def updatePoseCfg(self, posecfg_path, project_path=None, init_weights=None):
        # Updates pose_cfg.yaml with arguments (if supplied).
        pose_cfg = ruamel.yaml.load(open(posecfg_path))
        if project_path:
            pose_cfg['project_path'] = project_path
            print("Updated project_path in "+str(posecfg_path))
        if init_weights:
            pose_cfg['init_weights'] = init_weights
            print("Updated init_weights in "+str(posecfg_path))
        with open(posecfg_path, 'w') as file:
            ruamel.yaml.round_trip_dump(pose_cfg, file)

    def findBlanks(self, files, return_indices=False):
        print("Finding mismatched frames...")
        pngs = [file for file in files if os.path.splitext(file)[1]=='.png']
        filesizes = [os.stat(file).st_size for file in pngs]
        z = stats.zscore(filesizes)
        dictionary = {k:[v1,v2] for (k,v1,v2) in zip(pngs, filesizes, z)}
        blanks = [key for key in dictionary.keys() if dictionary[key][1]<0]
        print("Found ",str(len(blanks))," mismatched frames.")
        if return_indices:
            result = [os.path.splitext(os.path.basename(file))[0][3:].lstrip('0') for file in blanks]
            return result
        else:
            return blanks

    def exciseRevise(self, event, indices):
        self.config = ruamel.yaml.load(open(self.yaml))
        files_to_delete = []
        all_deleted = []
        not_deleted = []
        for file in self.config['history'][event]['files']:
            if os.path.splitext(os.path.basename(file))[0][3:].lstrip('0') in indices:
                files_to_delete.append(file)
        files_to_delete_dict = {k:v for k, v in enumerate(files_to_delete)}
        print(files_to_delete_dict)
        exclusions = input("Proceed with deleting these files and erasing them from history? Hit enter to continue, type 'quit' to exit, or specify files to exclude by their numeric indices, separated by a space (e.g. 0 2)").rstrip(' ')
        if exclusions == 'quit':
            sys.exit("Pipeline terminated.")
        else:
            final_list = [files_to_delete_dict[key] for key in files_to_delete_dict.keys() if str(key) not in exclusions]
        for event in self.config['history']:
            deleted = []
            for file in self.config['history'][event]['files']:
                if file in final_list:
                    try:
                        os.remove(file)
                        print("deleted ", file, " and removed it from history")
                        deleted.append(file)
                    except:
                        not_deleted.append(file)
                        print("error while deleting",file)
            all_deleted.extend(deleted)
            new_files = [file for file in self.config['history'][event]['files'] if file not in deleted]
            self.config['history'][event]['files'] = new_files
        print("Redacted ",str(len(all_deleted))," files.")
        print("Failed to redact ",str(len(not_deleted))," files. Check output for details.")
        with open(self.yaml, 'w') as file:
            ruamel.yaml.round_trip_dump(self.config, file)
        return [new_files, not_deleted]



    def cleanup(self, events):
        # Deletes all files created by an operation, along with their immediate parent directory (if empty)
        self.config = ruamel.yaml.load(open(self.yaml))
        for event in events:
            if event not in self.config['history']:
                print('Not found in log')
                pass
            dirs = []
            files_deleted = []
            for path in self.config['history'][event]['files']:
                dir = os.path.dirname(path)
                if dir not in dirs:
                    dirs.append(dir)
                if os.path.exists(path):
                    files_deleted.append(path)
                    os.remove(path)
                else:
                    print("File not found: "+path)
            print("Deleted "+str(len(files_deleted))+" files associated with operation "+str(self.config['history'][event]['operation'])+" at "+event)
            for dir in dirs:
                if len(os.listdir(dir)) == 0:
                    try:
                        os.rmdir(dir)
                        print("Deleted directory ", dir)
                    except:
                        print( "FAILED :", dir )
                        pass
            self.config['history'].pop(event, None)
        with open(self.yaml, 'w') as file:
            ruamel.yaml.round_trip_dump(self.config, file)

    def evaluateAndAnalyze(self, shuffle=1, trainingsetindex=0):
        trainposeconfigfile, testposeconfigfile, snapshotfolder = self.dlc.return_train_network_path(self.yaml, shuffle, self.config['TrainingFraction'][trainingsetindex])
        self.plotLoss(os.path.join(snapshotfolder,'learning_stats.csv'))
        evaluation_return = self.updateWithFunc('evaluation', self.trackFiles, self.dirs['evaluation'],
                                                self.dlc.evaluate_network,
                                                config=self.yaml,
                                                Shuffles=[shuffle],
                                                trainingsetindex=trainingsetindex,
                                                plotting=True,
                                                gputouse=0
                                                )
        analysis_return = self.updateWithFunc('analysis', self.trackFiles,
                                                os.path.dirname(self.vids_merged[0]),
                                                self.dlc.analyze_videos,
                                                config=self.yaml,
                                                videos=self.vids_merged,
                                                shuffle=shuffle,
                                                trainingsetindex=trainingsetindex,
                                                gputouse=0,
                                                save_as_csv=True
                                                )

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

    def getLatestSnapshot(self, snapshot_dir):
      snapshots = []
      for root, dirs, files in os.walk(snapshot_dir):
        for name in files:
          if name.split('-')[0]=='snapshot':
            snapshots.append(name.split('.')[0])
      uniques = {}
      uniques = {index for index in snapshots if index not in uniques}
      sorted_uniques = sorted(uniques, key=lambda index: index.split('-')[1], reverse=True)
      latest = os.path.join(snapshot_dir, sorted_uniques[0])
      return latest

    def vidToPngs(self, video_path, output_dir=None, indices_to_match=[], name_from_folder=True, compression=0):
        # Takes a list of frame numbers and exports matching frames from a video as pngs. Optionally, compress the output PNGs. Factor ranges from 0 (no compression) to 9 (most compression)
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
            os.makedirs(out_dir)
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
                    if self.in_colab:
                        plt.imshow(frame)
                        plt.show()
                    else:
                        cv2.imshow('frame',frame)
                    cv2.imwrite(png_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, compression])
                    frame_counter = 0
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    frame_index += 1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Finished extracting .pngs from "+str(os.path.basename(video_path)))
        return png_list

    def changeFileName(self, file_path, insertion, new_extension=None, pos=0):
        # Given any filepath, inserts a custom string insertion at position pos. Default behavior is prefix.
        location, name = os.path.split(file_path)
        left, right = name[:pos], name[pos:]
        modified_name = left + insertion + right
        if new_extension != None:
            temp_name, old_ext = os.path.splitext(modified_name)
            modified_name = temp_name + new_extension
        result = os.path.join(location, modified_name)
        return result

    def bakeMetadata(self, input_path, output_path, codec='avc1'):
        # Bakes experiment metadata (Experiment day, trial condition, frame number) into each video frame. Expects the following folder structure: `[experiment]/[trial]/[video]`.
        def getImmediateAncestors(file_path, depth=1):
            # Returns a list of a given path's immediate parent directories, in ascending order. Level is specified as `depth`.
            result = []
            for i in range(depth):
                parent_path, child_name = os.path.split(file_path)
                file_path = parent_path
                result.append(child_name)
            return result
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = round(cap.get(5),2)
        pos_x = round(frame_width/50)
        off_y = round(frame_height/50)
        off_y_initial = off_y
        frame_index = 1
        metadata = getImmediateAncestors(input_path, 3)[1:]
        metadata.append(frame_index)
        font_family = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.4
        font_color = (255, 255, 255)
        if codec == 'uncompressed':
            pix_format = 'gray'   ##change to 'yuv420p' for color or 'gray' for grayscale. 'pal8' doesn't play on macs
            p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)), '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), output_video], stdin=PIPE)
        else:
            if codec == 0:
                fourcc = 0
            else:
                fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path,
                                  fourcc,
                                  frame_rate,(frame_width, frame_height))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                off_y = off_y_initial
                for metadatum in metadata:
                    pos_y = frame_height-off_y
                    frame = cv2.putText(frame, str(metadatum), (pos_x, pos_y), font_family,
                                        font_size, font_color)
                    off_y = off_y+off_y_initial
                cv2.imshow('frame',frame)
                frame_index += 1
                metadata.pop()
                metadata.append(frame_index)
                if codec == 'uncompressed':
                    im = Image.fromarray(frame)
                    im.save(p.stdin, 'PNG')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        breakw
                else:
                    out.write(frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        if codec == 'uncompressed':
            p.stdin.close()
            p.wait()
        cap.release()
        if codec != 'uncompressed':
            out.release()
        cv2.destroyAllWindows()
        print("done!")
        return output_path


    def sortByCameraID(self, path_list, prefixes=['Cam','C00'], number_of_cameras=2):
        # Given a list of file paths, returns a list of lists sorted by camera identifier (in the format prefix->camera#).
        triaged_lists = [[] for cameras in range(number_of_cameras)]
        for i, triaged_list in enumerate(triaged_lists, start=1):
            for path in path_list:
                for prefix in prefixes:
                    match_string = prefix+str(i)
                    if re.search(match_string, path):
                        triaged_list.append(path)
                        break
        return triaged_lists

    def concatenateVideos(self, path_list, output_path, codec='avc1', interval=1):
        # Given a list of video paths, concatenates them into one long video. Passing in an optional downsampling factor tells the function to only capture one in every n frames.
        frame_index = 0
        video_index = 0
        cap = cv2.VideoCapture(path_list[0])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = round(cap.get(5),2)/interval
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path,
                              fourcc,
                              frame_rate,(frame_width, frame_height))
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame_index += 1
            if frame is None:
                print("end of video " + str(video_index) + " ... next one now")
                video_index += 1
                if video_index >= len(path_list):
                    break
                cap = cv2.VideoCapture(path_list[ video_index ])
                frame_index = 0
            elif frame_index == interval:
                frame = frame.astype(np.uint8)
                cv2.imshow('frame',frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_index = 0
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("done!")
        return output_path


    def mergeRGB(self, video_dict, output_path, codec='avc1', mode=None):
        # Takes a dictionary containing two video paths in the format `{'A':[path A], 'B':[path B]}` and exports a single new video with video A written to the red channel and video B written to the green channel. The blue channel is, depending on the value passed as "mode", either the difference blend between A and B, the multiply blend, or just a black frame. Output_path must contain extension.
        capA = cv2.VideoCapture(video_dict['A'])
        capB = cv2.VideoCapture(video_dict['B'])
        frame_width = int(capA.get(3))
        frame_height = int(capA.get(4))
        frame_rate = round(capA.get(5),2)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path,
                             fourcc,
                             frame_rate,(frame_width, frame_height))
        while(capA.isOpened()):
            retA, frameA = capA.read()
            retB, frameB = capB.read()
            if retA == True:
                frameA = cv2.cvtColor(frameA, cv2.COLOR_BGR2BGRA,4).astype(np.float32)
                frameB = cv2.cvtColor(frameB, cv2.COLOR_BGR2BGRA,4).astype(np.float32)
                frameA = cv2.normalize(frameA, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                frameB = cv2.normalize(frameB, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                if mode == "difference":
                    extraChannel = blend_modes.difference(frameA,frameB,1)
                elif mode == "multiply":
                    extraChannel = blend_modes.multiply(frameA,frameB,1)
                else:
                    extraChannel = np.zeros((frame_width, frame_height,3),np.uint8)
                    extraChannel = cv2.cvtColor(extraChannel, cv2.COLOR_BGR2BGRA,4).astype(np.float32)
                frameA = cv2.cvtColor(frameA, cv2.COLOR_BGRA2BGR).astype(np.uint8)
                frameB = cv2.cvtColor(frameB, cv2.COLOR_BGRA2BGR).astype(np.uint8)
                extraChannel = cv2.cvtColor(extraChannel, cv2.COLOR_BGRA2BGR).astype(np.uint8)
                frameA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
                frameB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
                extraChannel = cv2.cvtColor(extraChannel, cv2.COLOR_BGR2GRAY)
                merged = cv2.merge((extraChannel, frameB, frameA))
                cv2.imshow('merged',merged)
                out.write(merged)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        capA.release()
        capB.release()
        out.release()
        cv2.destroyAllWindows()
        print("done!")
        return output_path


    def splitRGB(self, input_path, codec='avc1'):
        # Takes a RGB video with different grayscale data written to the R, G, and B channels and splits it back into its component source videos.
        out_name = os.path.splitext(os.path.basename(input_path))[0]+'_split_'
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = round(cap.get(5),2)
        if codec == 'uncompressed':
            pix_format = 'gray'   ##change to 'yuv420p' for color or 'gray' for grayscale. 'pal8' doesn't play on macs
            p1 = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)), '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), out_name+'c1.avi'], stdin=PIPE)
            p2 = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)), '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), out_name+'c2.avi'], stdin=PIPE)
        else:
            if codec == 0:
                fourcc = 0
            else:
                fourcc = cv2.VideoWriter_fourcc(*codec)
            out1 = cv2.VideoWriter(out_name+'c1.mp4',
                                  fourcc,
                                  frame_rate,(frame_width, frame_height))
            out2 = cv2.VideoWriter(out_name+'c2.mp4',
                                  fourcc,
                                  frame_rate,(frame_width, frame_height))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                B, G, R = cv2.split(frame)
                cv2.imshow('frame',R)
                if codec == 'uncompressed':
                    imR = Image.fromarray(R)
                    imG = Image.fromarray(G)
                    imR.save(p1.stdin, 'PNG')
                    imG.save(p2.stdin, 'PNG')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    out1.write(R)
                    out2.write(G)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        if codec == 'uncompressed':
            p1.stdin.close()
            p1.wait()
            p2.stdin.close()
            p2.wait()
        cap.release()
        if codec != 'uncompressed':
            out1.release()
            out2.release()
        cv2.destroyAllWindows()
        print("done!")
        return [out_name+'c1.mp4', out_name+'c2.mp4']

    def getTimeStamp(self):
        ts = datetime.datetime.now().strftime("%d%b%y_%Hh%Mm%Ss")
        return ts

    def migrateProject(self, project_path=None, video_dir=None, init_weights=None, convert=False, win2unix = False):
        self.updateConfig()
        if not project_path:
            project_path = input("Enter the destination project path. This is the folder where config.yaml will live").strip('"')
        if not video_dir:
            video_dir = input("Enter the path to the destination video folder.").strip('"')
        if not init_weights:
            init_weights = input("Enter the path to the default weights file you want to use, e.g. {somepath}resnet_v1_50.ckpt").strip('"')
        video_sets = [os.path.join(video_dir, os.path.basename(video)) for video in self.config['video_sets']]
        if convert:
            h5_labeled_list = self.scanDir(self.dirs['labeled'], extension='h5', verbose=True)
            h5_training_list = self.scanDir(self.dirs['training'], extension='h5', verbose=True)
            h5_list = h5_labeled_list + h5_training_list
            if win2unix:
                for h5 in h5_list:
                    df = pd.read_hdf(h5, "df_with_missing")
                    df.index = df.index.str.replace("\\", "/")
                    df.to_hdf(h5, "df_with_missing", format="table", mode="w")
                print("Converted paths in "+str(len(h5_list))+"h5 annotation files from windows to unix format")
                project_path = project_path.replace("\\", "/")
                init_weights = init_weights.replace("\\", "/")
                video_sets = [ video.replace("\\", "/") for video in video_sets]
                print("Converted all yaml paths from windows to unix format")
            else:
                for h5 in h5_list:
                    df = pd.read_hdf(h5, "df_with_missing")
                    df.index = df.index.str.replace("/", "\\")
                    df.to_hdf(h5, "df_with_missing", format="table", mode="w")
                print("Converted paths in "+str(len(h5_list))+"h5 annotation files from unix to windows format")
                project_path = project_path.replace("/", "\\")
                init_weights = init_weights.replace("/", "\\")
                video_sets = [ video.replace("/", "\\") for video in video_sets]
                print("Converted all yaml paths from unix to windows format")
        self.updateConfig(project_path=project_path, videos=video_sets)
        pose_cfgs_list = self.scanDir(self.dirs['models'], extension="yaml", filters=["pose_cfg"], filter_out=False, verbose=True)
        for pose_cfg_file in pose_cfgs_list:
            self.updatePoseCfg(pose_cfg_file, project_path=project_path, init_weights=init_weights)
        print("Successfully migrated project")
        return


    def filterByExtension(self, list, extension):
        filtered_list = [path for path in list if extension in os.path.splitext(path)[1]]
        return filtered_list

    def matchFrames(self, list):
        # Recurses through a list of paths looking for unique frame numbers, returns a list of indices.
        frame_list = self.filterByExtension(list, extension = 'png')
        extracted_indices = [int(os.path.splitext(os.path.basename(png))[0][3:].lstrip('0')) for png in frame_list]
        unique_indices = {}
        unique_indices = {index for index in extracted_indices if index not in unique_indices}
        result = sorted(unique_indices)
        print("Found "+str(len(result))+" unique frame indices")
        return result

    def extractMatchedFrames(self, indices, output_dir, src_vids=[], folder_suffix=None, timestamp=False, compression=0):
        # Given a list of frame indices and a list of source videos, produces one folder of matching frame pngs per source video. Optionally, compress the output PNGs. Factor ranges from 0 (no compression) to 9 (most compression)
        extracted_frames = []
        if timestamp:
            ts = '_'+self.getTimeStamp()
        else:
            ts = ''
        for video in src_vids:
            if folder_suffix:
                out_name = os.path.splitext(os.path.basename(video))[0]+folder_suffix+ts
            else:
                out_name = os.path.splitext(os.path.basename(video))[0]+ts
            output = os.path.join(output_dir,out_name)
            frames_from_vid = self.vidToPngs(video, output, indices_to_match=indices, name_from_folder=False, compression=compression)
            extracted_frames.append(frames_from_vid)
        combined_list = [y for x in extracted_frames for y in x]
        print("Extracted "+str(len(indices))+" matching frames from each of "+str(len(src_vids))+" source videos")
        return combined_list

    def importXma(self, event, csv_path=None, outlier_mode=False, indices_to_drop=[], swap=False, cross=False):
        # Interactively imports labels from XMALab by substituting frames from merged video for original raw frames. Updates config.yaml to point to substituted video.
        if not csv_path:
            csv_path = input("Enter the full path to XMALab 2D XY coordinates csv, or type 'quit' to abort.").strip('"')
        if csv_path == "quit":
            sys.exit("Pipeline terminated.")
        indices_to_import = self.matchFrames(self.config['history'][event]['files'])
        if indices_to_drop:
            print(indices_to_import)
            csv_indices = range(len(indices_to_import))
            print(csv_indices)
            index_dict = dict(zip(indices_to_import, csv_indices))
            indices_to_import = list(sorted(set(indices_to_import).difference(set(indices_to_drop))))
            csv_indices_to_import = [index_dict[x] for x in indices_to_import]
            print(csv_indices_to_import)
            print(indices_to_import)
            df=pd.read_csv(csv_path,sep=',',header=0,dtype='float')
            backup_path = os.path.join(os.path.dirname(csv_path),os.path.splitext(os.path.basename(csv_path))[0]+'_with_undigitizable.csv')
            df.to_csv(backup_path,index=False)
            df1=df.iloc[csv_indices_to_import]
            df1.to_csv(csv_path,index=False)
            ts = self.getTimeStamp()
            update = {ts:{'operation':'dropUndigitizableFrames','files':indices_to_drop}}
            self.updateConfig(event=update)
        spliced_markers = self.spliceXma2Dlc(self.vids_merged[0], csv_path, indices_to_import, outlier_mode, swap, cross)
        self.extractMatchedFrames(indices_to_import, output_dir=self.dirs['labeled'], src_vids=self.vids_merged)
        self.updateConfig(videos=self.vids_merged, bodyparts=spliced_markers)
        if outlier_mode:
            self.deleteLabeledFrames(self.dirs['labeled'])
            self.dlc.merge_datasets(self.yaml)
            self.mergeOutliers()
            self.dlc.create_training_dataset(self.yaml)
            # maybe track and print? not sure yet
        else:
            self.dlc.create_training_dataset(self.yaml)


    def mergeOutliers(self):
        # Merges CollectedData_[experimenter].h5 with MachineLabelsRefine.h5. Renames original CollectedData_[experimenter].h5 and sets it aside.
        df0 = pd.read_hdf(os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".h5")),"df_with_missing")
        df1 = pd.read_hdf(os.path.join(self.dirs['spliced'],("MachineLabelsRefine.h5")),"df_with_missing")
        df_combined = pd.concat([df0, df1])
        df_combined.sort_index(inplace=True)
        df_combined.to_hdf(
            os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".h5")),
            key="df_with_missing",
            format="table",
            mode="w"
        )
        df_combined.to_csv(
            os.path.join(self.dirs['spliced'],("CollectedData_"+self.experimenter+".csv"))
        )
        print("Merged imported outliers")

    def deleteLabeledFrames(self, dir):
        frame_list = self.scanDir(dir, extension='png', filters=['labeled'], filter_out=False)
        if not frame_list:
            print("No labeled frames found!")
        for frame in frame_list:
            os.remove(frame)
        print("Deleted "+str(len(frame_list))+" DLC preview frames")

    def swapFeaturesHdf(self, hdf_path, swap: bool, cross: bool):
        swap_name = '_swap' if swap else ''
        cross_name = '_cross' if cross else ''
        new_name = swap_name+cross_name
        df = pd.read_hdf(hdf_path, "df_with_missing")
        df = df.reset_index().melt(id_vars=['index'])
        names_initial = df.bodyparts.unique()
        parts_initial = [name.rsplit('_',1)[0] for name in names_initial]
        parts_unique_initial = []
        for part in parts_initial:
            if not part in parts_unique_initial:
                parts_unique_initial.append(part)
        bodyparts_XY=[]
        for part in names_initial:
                    bodyparts_XY.append(part+'_X')
                    bodyparts_XY.append(part+'_Y')
        df['id'] = df['bodyparts']+'_'+df['coords'].str.upper()
        df = df.pivot(index='index',columns='id',values='value')
        extracted_frames = [index.split('/',2)[-1] for index in df.index]
        df = df.reindex(columns=bodyparts_XY)
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
            parts_postswap = df.columns.tolist()
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
            parts_postcross = df.columns.tolist()
            print(crosses)
        names_final = df.columns.values
        parts_final = [name.rsplit('_',1)[0] for name in names_final]
        parts_unique_final = []
        for part in parts_final:
            if not part in parts_unique_final:
                parts_unique_final.append(part)
        df['scorer']=self.experimenter
        df['frame_index']=df.index
        df = df.melt(id_vars=['frame_index','scorer'])
        new = df['variable'].str.rsplit("_",n=1,expand=True)
        df['variable'],df['coords'] = new[0], new[1]
        df=df.rename(columns={'variable':'bodyparts'})
        df['coords']=df['coords'].str.rstrip(" ").str.lower()
        cat_type = pd.api.types.CategoricalDtype(categories=parts_unique_final,ordered=True)
        df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype(cat_type)
        newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
        newdf.index.name=None
        ts = self.getTimeStamp()
        data_name = os.path.join(os.path.dirname(hdf_path),(os.path.splitext(os.path.basename(hdf_path))[0]+new_name+'.h5'))
        tracked_hdf = os.path.join(os.path.dirname(hdf_path),(os.path.splitext(os.path.basename(hdf_path))[0]+new_name+ts+'.h5'))
        newdf.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
        newdf.to_hdf(tracked_hdf, 'df_with_missing', format='table', mode='w')
        tracked_csv = data_name.split('.h5')[0]+'_'+ts+'.csv'
        newdf.to_csv(tracked_csv)
        tracked_files = [tracked_hdf, tracked_csv]
        self.updateWithFiles(new_name[1:], tracked_files)
        print("Successfully generated synthetic features; saved "+str(data_name)+", "+str(tracked_hdf)+", and "+str(tracked_csv))
        return parts_unique_final

    def swapCrossPoseCfgs(self, train_yaml, test_yaml, bodyparts):
        ts = self.getTimeStamp()
        new_train = os.path.join(os.path.dirname(train_yaml),(os.path.splitext(os.path.basename(train_yaml))[0]+'_swcx_'+ts+'.yaml'))
        new_test = os.path.join(os.path.dirname(test_yaml),(os.path.splitext(os.path.basename(test_yaml))[0]+'_swcx_'+ts+'.yaml'))
        new_config = os.path.join(os.path.dirname(self.yaml),(os.path.splitext(os.path.basename(self.yaml))[0]+'_swcx_'+ts+'.yaml'))
        shutil.copyfile(train_yaml, new_train)
        shutil.copyfile(test_yaml, new_test)
        shutil.copyfile(self.yaml, new_config)
        train_file = ruamel.yaml.load(open(new_train))
        test_file = ruamel.yaml.load(open(new_test))
        config_file = ruamel.yaml.load(open(new_config))
        all_joints_formatted = [[i] for i in range(len(bodyparts))]
        train_file['all_joints_names'] = bodyparts
        test_file['all_joints_names'] = bodyparts
        train_file['num_joints'] = len(bodyparts)
        test_file['num_joints'] = len(bodyparts)
        config_file['bodyparts'] = bodyparts
        train_file['all_joints'] = all_joints_formatted
        test_file['all_joints'] = all_joints_formatted
        with open(new_train, 'w') as file:
            ruamel.yaml.round_trip_dump(train_file, file)
        print('saved '+new_train)
        with open(new_test, 'w') as file:
            ruamel.yaml.round_trip_dump(test_file, file)
        print('saved '+new_test)
        with open(new_config, 'w') as file:
            ruamel.yaml.round_trip_dump(config_file, file)
        print('saved '+new_config)


    def spliceXma2Dlc(self, substitute_video, csv_path, frame_indices, outlier_mode=False, swap=False, cross=False):
        # Takes csv of XMALab 2D XY coordinates from 2 cameras, outputs spliced hdf+csv data for DeepLabCut
        substitute_name = os.path.splitext(os.path.basename(substitute_video))[0]
        substitute_data_relpath = os.path.join("labeled-data",substitute_name)
        substitute_data_abspath = os.path.join(self.wd,substitute_data_relpath)
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
        df['scorer']=self.experimenter
        df = df.melt(id_vars=['frame_index','scorer'])
        new = df['variable'].str.rsplit("_",n=1,expand=True)
        df['variable'],df['coords'] = new[0], new[1]
        df=df.rename(columns={'variable':'bodyparts'})
        df['coords']=df['coords'].str.rstrip(" ").str.lower()
        cat_type = pd.api.types.CategoricalDtype(categories=parts_unique_final,ordered=True)
        df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype(cat_type)
        newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
        newdf.index.name=None
        ts = self.getTimeStamp()
        if not os.path.exists(substitute_data_abspath):
            os.makedirs(substitute_data_abspath)
        if outlier_mode:
            data_name = os.path.join(substitute_data_abspath,"MachineLabelsRefine.h5")
            tracked_hdf = os.path.join(substitute_data_abspath,("MachineLabelsRefine_"+ts+".h5"))
        else:
            data_name = os.path.join(substitute_data_abspath,("CollectedData_"+self.experimenter+".h5"))
            tracked_hdf = os.path.join(substitute_data_abspath,("CollectedData_"+self.experimenter+"_"+ts+".h5"))
        newdf.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
        newdf.to_hdf(tracked_hdf, 'df_with_missing', format='table', mode='w')
        tracked_csv = data_name.split('.h5')[0]+'_'+ts+'.csv'
        newdf.to_csv(tracked_csv)
        tracked_files = [tracked_hdf, tracked_csv]
        self.updateWithFiles('spliceXma2Dlc', tracked_files)
        print("Successfully spliced XMALab 2D points to DLC format; saved "+str(data_name)+", "+str(tracked_hdf)+", and "+str(tracked_csv))
        return parts_unique_final

    def filter_by_likelihood(self, df, p_cutoff):
        scorer, bodyparts = list(df.columns.levels[0])[0], list(df.columns.levels[1])
        for part in bodyparts:
            df.loc[:,(scorer,part,'x')].mask(df.loc[:,(scorer,part,'likelihood')] < p_cutoff, inplace=True)
            df.loc[:,(scorer,part,'y')].mask(df.loc[:,(scorer,part,'likelihood')] < p_cutoff, inplace=True)
        return df

    def splitDlc2Xma(self, hdf_path, bodyparts, frame_paths=[], interval=1, likelihood_threshold=0):
        p_string = '_p'+str(likelihood_threshold)+'_' if likelihood_threshold else ''
        interval_string = 'every'+str(interval) if interval != 1 else ''
        bodyparts_XY = []
        ts = self.getTimeStamp()
        for part in bodyparts:
            bodyparts_XY.append(part+'_X')
            bodyparts_XY.append(part+'_Y')
        df=pd.read_hdf(hdf_path)
        df=df[::interval]
        if frame_paths:
            def get_index(path):
                [png_parent, png] = os.path.split(path)
                [run_parent, run] = os.path.split(png_parent)
                [base_parent, base] = os.path.split(run_parent)
                result = os.path.join(base, run, png)
                return result
            pngs = [get_index(path) for path in frame_paths if os.path.splitext(path)[1]=='.png']
            unique_pngs = {}
            unique_pngs = {png for png in pngs if png not in unique_pngs}
            sorted_pngs = sorted(unique_pngs)
            df=df.loc[df.index.intersection(sorted_pngs)]
        if likelihood_threshold:
            df = self.filter_by_likelihood(df, likelihood_threshold)
        df = df.reset_index().melt(id_vars=['index'])
        df = df[df['coords'] != 'likelihood']
        df['id'] = df['bodyparts']+'_'+df['coords'].str.upper()
        df[['index','value','id']]
        df = df.pivot(index='index',columns='id',values='value')
        if type(df.index[0]) == str:
            extracted_frames = [index.split('\\')[-1] for index in df.index]
        else:
            extracted_frames = list(df.index)
        df = df.reindex(columns=bodyparts_XY)
        tracked_csv = os.path.splitext(hdf_path)[0]+'_split_'+p_string+ts+'.csv'
        df.to_csv(tracked_csv,index=False)
        self.updateWithFiles('splitDlc2Xma', [tracked_csv])
        print("Successfully split DLC format to XMALab 2D points; saved "+str(tracked_csv))
        return extracted_frames

    def getBodypartsFromXmaExport(self, csv_path):
    	df = pd.read_csv(csv_path, sep=',',header=0, dtype='float',na_values='NaN')
    	names = df.columns.values
    	parts = [name.rsplit('_',2)[0] for name in names]
    	parts_unique = []
    	for part in parts:
    		if not part in parts_unique:
    			parts_unique.append(part)
    	return parts_unique

    def plotLoss(self, learning_stats_path):
        df=pd.read_csv(learning_stats_path,sep=',',header=None,names=['iteration','loss','lr'],dtype='float',na_values='NaN')
        fig, ax1 = plt.subplots(1, 1, figsize=(8,4))
        ax1.plot(df['iteration'],df['loss'],'b')
        ax2=ax1.twinx()
        ax2.plot(df['iteration'],df['lr'],'g')
        ax1.set_xlabel(r'Iteration', fontsize=10)
        ax1.set_ylabel(r'Loss', fontsize=10, color='blue')
        ax2.set_ylabel(r'Learning rate', fontsize=10, color='green')
        ax1.tick_params(labelsize=10)
        ax2.tick_params(labelsize=10)
        fig.tight_layout()
        plt.show()
        return fig
