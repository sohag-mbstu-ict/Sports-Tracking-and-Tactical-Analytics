import os
import cv2
from typing import Optional, List
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from tinydb import TinyDB, Query
import pandas as pd
import torch
from sports.common.team import TeamClassifier
from team_classifier import team_classification

class player_goalkeeper_referee_detection:
    def __init__(self,ball_player_goalkeeper_referee_det_model):
        # Load models
        self.ball_player_goalkeeper_referee_det_model = ball_player_goalkeeper_referee_det_model
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3
        self.team_classifier = TeamClassifier(device="cuda")
        self.team_clssifier_obj = team_classification()
        self.ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),thickness=2)
        self.label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                             text_color=sv.Color.from_hex('#000000'),text_position=sv.Position.BOTTOM_CENTER)
        self.triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'),base=25,height=21,outline_thickness=1)
        
        
    def resolve_goalkeepers_team_id(self, players: sv.Detections,
        goalkeepers: sv.Detections
    ) -> np.ndarray:
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        return np.array(goalkeepers_team_id)

    def mapping_det_bbox_with_track_ID(self,detections):
        bboxes = detections.xyxy
        tracker_ids = detections.tracker_id
        tracker_id_index = 0
        centers_dict = {}
        bbox_dict = {}
        # Loop through detections
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx = x1 + (x2 - x1) * 0.5
            cy = y1 + (y2 - y1) * 0.5
            
            centers_dict[int(tracker_ids[tracker_id_index])] = [round(float(cx),2), round(float(cy),2)]
            bbox_dict[int(tracker_ids[tracker_id_index])] = bbox
            tracker_id_index += 1
        # Sort dictionary by key
        sorted_centered_dict = dict(sorted(centers_dict.items()))
        sorted_bbox_dict = dict(sorted(bbox_dict.items()))
        return sorted_centered_dict, sorted_bbox_dict
        
    def ball_player_goalkeeper_referee_detection(self,frame,tracker,df,frame_count):
        detections = self.ball_player_goalkeeper_referee_det_model(frame,imgsz = 1280)[0]
        detections = sv.Detections.from_ultralytics(detections)

        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != self.BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[all_detections.class_id == self.GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == self.PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == self.REFEREE_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        
        players_detections.class_id = self.team_classifier.predict(players_crops)
        goalkeepers_detections.class_id = self.resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections)
        referees_detections.class_id -= 1
        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections])

        # mappint label ID to color ID; [0,1 -> player  2 -> referee]
        all_detections.class_id = all_detections.class_id.astype(int)
        tracID_to_colorID = dict(zip(all_detections.tracker_id.tolist(),all_detections.class_id.tolist()))
        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id]

        annotated_frame = frame.copy()
        # annotated_frame = self.ellipse_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=all_detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels)
        annotated_frame = self.triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)
                
        goalkeepers_centered_with_id = None
        players_centered_with_id     = None
        referees_centered_with_id    = None
        goalkeepers_bbox_with_id = None
        players_bbox_with_id     = None
        referees_bbox_with_id    = None
        
        if len(goalkeepers_detections)>0 :
            goalkeepers_centered_with_id, goalkeepers_bbox_with_id = self.mapping_det_bbox_with_track_ID(goalkeepers_detections)
        if len(players_detections)>0 :   
            players_centered_with_id, players_bbox_with_id = self.mapping_det_bbox_with_track_ID(players_detections)
        if len(referees_detections)>0 :   
            referees_centered_with_id, referees_bbox_with_id = self.mapping_det_bbox_with_track_ID(referees_detections)
        
        # annotated_frame, class_id_list = self.team_clssifier_obj.team_classifier(frame, players_bbox_with_id, 
        #                                 referees_detections, goalkeepers_detections, ball_detections) # Function Call
        
        df.loc[frame_count] = [frame_count,goalkeepers_centered_with_id,players_centered_with_id,players_bbox_with_id,referees_centered_with_id,tracID_to_colorID,ball_detections.xyxy]
        return df, annotated_frame
    
   