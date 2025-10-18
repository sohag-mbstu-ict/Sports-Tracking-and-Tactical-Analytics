import os
import cv2
from typing import Optional, List
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, Counter
from tinydb import TinyDB, Query
from playerGoalkeeperReferee import player_goalkeeper_referee_detection
from field_keypoints_det import Field_Keypoints_Extractor
from kalmanFilter import smoothed_position_using_kalman_filter
from FieldVisualizer import FootballFieldVisualizer
import pandas as pd


class FootballAIPipeline:
    def __init__(self):
        # Load models
        self.detection_model_for_keypoints = YOLO("/media/gflml/Volume E/yolo/Soccer/trained_model/field_keypoints_det_130.pt")
        self.ball_player_goalkeeper_referee_det_model = YOLO("/media/gflml/Volume E/yolo/Soccer/trained_model/Player_Referee_Goalkeeper_Detection.pt")
        self.playerGoalkeeperReferee_obj = player_goalkeeper_referee_detection(self.ball_player_goalkeeper_referee_det_model) # class instantiate
        self.kalman_filter_obj = smoothed_position_using_kalman_filter() # class instantiate
        self.field_points_obj = Field_Keypoints_Extractor(self.detection_model_for_keypoints) # class instantiate
        self.field_visualizer_obj = FootballFieldVisualizer() # class instantiate
        self.player_goalkeeper_referee_df = pd.DataFrame(columns=['Frame_Num','Goalkeepers','Players','Players_bbox','Referees','team_classifier','Ball'])
        self.field_keypoints_df = pd.DataFrame(columns=['Frame_Num','frame_reference_points','keypoints_with_id'])
        
        
    def get_video_writer(self, video_name, frame_size=(1280, 720), fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = f"/media/gflml/Volume E/yolo/Soccer/output/{video_name}.mp4"
        return cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)


    def get_majority_team_classifier(self, player_goalkeeper_referee_df: pd.DataFrame, column_name: str = "team_classifier") -> dict:
        freqs = defaultdict(Counter)
        # Count frequencies for each ID across all dictionaries
        for d in player_goalkeeper_referee_df[column_name]:
            for k, v in d.items():
                freqs[k][v] += 1
        # Pick majority class for each ID
        majority_dict = {k: freqs[k].most_common(1)[0][0] for k in freqs}
        # Convert dict to DataFrame
        # Step 2: Replace the column with this dictionary in all rows
        player_goalkeeper_referee_df["team_classifier"] = [majority_dict] * len(player_goalkeeper_referee_df)
        return player_goalkeeper_referee_df

        
   
    def run_pipeline(self):
        # input_video_path = "/media/mtl/Volume F/PROJECTS/projects/Soccer/input_vid/Untitled-Video-2.mp4"
        input_video_path = "/media/gflml/Volume E/yolo/Soccer/input_vid/0bfacc_0.mp4"
        players_referees_goalkeepers_df = pd.read_excel("/media/gflml/Volume E/yolo/Soccer/player_goalkeeper_referee_df.xlsx")
        video_writer = self.get_video_writer("det",(1280, 720))
        # players_referees_goalkeepers_df = pd.read_excel("/media/mtl/Volume F/PROJECTS/projects/Soccer/team_classifier_smoothed.xlsx")
        cap = cv2.VideoCapture(input_video_path)
        frame_count = 0  # Counter for frames 
        tracker = sv.ByteTrack()
        tracker.reset()       
        while cap.isOpened():
            frame_count+=1
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends
            if frame_count%10 == 0:
                print("----------------- frame_count ---------------- : ",frame_count)
            # if(frame_count<230):
            #     continue
            if(frame_count>100):
                break
            
            # ------------------------------------------- Get 32 field keypoints in excel file --------------------------------------------
            # field_keypoints detection
            field_keypoints_df = self.field_points_obj.extract_frame_and_pitch_points_using_detection_model(
                                            frame, frame_count, self.field_keypoints_df)
        
        # reset so that the index will start from 0, will not show key err
        field_keypoints_df = field_keypoints_df.reset_index(drop=True)
        # remove False positives field keypoints detection
        field_keypoints_df = self.field_points_obj.remove_false_positive_field_keypoints(
                                                       field_keypoints_df)
        
        # reset so that the index will start from 0, will not show key err
        field_keypoints_df = field_keypoints_df.reset_index(drop=True)
        self.field_visualizer_obj.main_func_to_draw_32_football_field_keypoints(input_video_path, field_keypoints_df, 
                                            players_referees_goalkeepers_df)
            
        # Example: save dataframe to Excel
        field_keypoints_df.to_excel("field_keypoints_df.xlsx", index=False, engine='openpyxl')
        # ------------------------------------------- Get player_goalkeeper_referee_df excel file --------------------------------------------
        
        
        
        
        #     # ------------------------------------------- Get player_goalkeeper_referee_df excel file --------------------------------------------
        #     # ball, player, goalkeeper, referee detection
        #     player_goalkeeper_referee_df,det_frame = self.playerGoalkeeperReferee_obj.ball_player_goalkeeper_referee_detection(
        #                                                                     frame,tracker,self.player_goalkeeper_referee_df,frame_count) 
        #     det_frame = cv2.resize(det_frame,(1280,720))
        #     video_writer.write(det_frame)
        # video_writer.release()
        # # df = self.kalman_filter_obj.handle_unexpected_bbox_size(df) # Function call (it will not work)
        # # player_goalkeeper_referee_df = self.kalman_filter_obj.smoothed_player_goalkeeper_referee_position_using_kalman_filter(player_goalkeeper_referee_df)
        # # Example: save dataframe to Excel
        # player_goalkeeper_referee_df = self.get_majority_team_classifier(player_goalkeeper_referee_df,"team_classifier")

        # player_goalkeeper_referee_df.to_excel("/home/gflml/yolo/Soccer/player_goalkeeper_referee_df.xlsx", index=False, engine='openpyxl')
        # # ------------------------------------------- Get player_goalkeeper_referee_df excel file --------------------------------------------
        
# ================= Usage =================
if __name__ == "__main__":
    pipeline = FootballAIPipeline()
    pipeline.run_pipeline()
    

        