import cv2
import json
import ast
import numpy as np
from filterpy.kalman import KalmanFilter
import supervision as sv
from ultralytics import YOLO
from tinydb import TinyDB, Query
from sports.configs.soccer import SoccerPitchConfiguration
from tinydb import TinyDB


class Field_Keypoints_Extractor:
    def __init__(self, keypoints_model_bbox):
        self.CONFIG = SoccerPitchConfiguration()
        self.detection_model_for_keypoints = keypoints_model_bbox
        
    def euclidean_distance(self,p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def remove_false_positive_field_keypoints(self,df):
        df["keypoints_with_id"] = df["keypoints_with_id"].apply(lambda x: ast.literal_eval(str(x)))
        cleaned_rows = []
        for i in range(len(df)):
            row_dict = df.loc[i, "keypoints_with_id"].copy()
            new_row = {}
            for track_id, point in row_dict.items():
                keep = False
                id_frequency = 0
                # Compare with next 5 rows
                for j in range(1, 6):
                    if i + j < len(df):
                        next_row = df.loc[i + j, "keypoints_with_id"]

                        if track_id in next_row:
                            if self.euclidean_distance(point, next_row[track_id]) < 35:
                                id_frequency += 1
                if id_frequency>4:
                    new_row[track_id] = point
            cleaned_rows.append(new_row)
        df["cleaned_keypoints"] = cleaned_rows
        return df

    def extract_frame_and_pitch_points_using_detection_model(self,frame, frame_count, field_keypoints_df):
        has_homography = True
        result = self.detection_model_for_keypoints.predict(frame, imgsz=1280, conf=0.35)
        bboxes = result[0].boxes.data
        centers_dict = {}
        # Loop through detections
        for bbox in bboxes:
            x1, y1, x2, y2, conf, class_id = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx = x1 + (x2 - x1) * 0.5
            cy = y1 + (y2 - y1) * 0.5
            # Store class_id → center mapping
            centers_dict[int(class_id)] = (cx, cy)

        # Sort dictionary by key
        keypoints_with_id = dict(sorted(centers_dict.items()))
        centers = [list(v) for v in keypoints_with_id.values()]
        frame_reference_points = centers
        field_keypoints_df.loc[frame_count] = [frame_count,frame_reference_points,keypoints_with_id]
        return field_keypoints_df
    
   