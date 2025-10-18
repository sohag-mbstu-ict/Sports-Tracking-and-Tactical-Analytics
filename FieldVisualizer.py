import os
import cv2
import re
from typing import Optional, List
import numpy as np
import supervision as sv
from tqdm import tqdm
import pandas as pd
import ast
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from kalmanFilter import smoothed_position_using_kalman_filter
from ultralytics import YOLO

class FootballFieldVisualizer:
    def __init__(self):
        self.player_trackers = {}
        self.CONFIG = SoccerPitchConfiguration()
        self.pitch = draw_pitch(self.CONFIG)
        self.kalman_filter_obj = smoothed_position_using_kalman_filter()
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)
        
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex('#FF1493'), radius=8)
        self.edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex('#00BFFF'), thickness=2, edges=self.CONFIG.edges)
        self.vertex_annotator_2 = sv.VertexAnnotator(color=sv.Color.from_hex('#00BFFF'), radius=8)


    def parse_numeric_lists_for_ball_detection(raw_string: str, missing_value: float = 0.0) -> list[list[float]]:
        try:
            # Replace any occurrence of ',,' or '[,' or ',]' with the missing_value
            cleaned_str = re.sub(r'(?<=\[|,)(?=,|])', str(missing_value), raw_string)
            # Safely evaluate string to Python object
            parsed_list = ast.literal_eval(cleaned_str)
            # Convert all inner values to floats
            result = [[float(x) for x in row] for row in parsed_list]
            return result
        except Exception as e:
            print(f"Error parsing string: {e}")
            return []  # fallback


    def draw_bbox_with_label(self, annotated_frame, frame_count, players_referees_goalkeepers_df):
        org_img = annotated_frame.copy()
        if frame_count in players_referees_goalkeepers_df["Frame_Num"].values:
            row = players_referees_goalkeepers_df.loc[players_referees_goalkeepers_df["Frame_Num"] == frame_count].iloc[0]
            Players  = row["Players"]  # this is already a dict
            Referees = row["Referees"]
            Goalkeepers = row["Goalkeepers"]
            ball=row["Ball"]
            # Remove outer brackets and extra spaces
            numbers_str = ball.strip()[2:-2].strip()  # remove '[[' and ']]'
            # Replace spaces between numbers with commas
            numbers_str = re.sub(r'\s+', ',', numbers_str)
            # Wrap back into list of list
            ball_fixed = f'[[{numbers_str}]]'
            # ball_bbox = self.parse_numeric_lists_for_ball_detection(ball_fixed) # Function Call
            try  :
                ball_bbox = ast.literal_eval(ball_fixed)
            except:
                ball_bbox= []
            
            if(pd.notna(Players)):
                Players = ast.literal_eval(Players)
                players_labels = list(Players.keys())
                players_bottom_centers = list(Players.values())
                track_id_index = 0
                smoothed_x_y_list = []
                for center in players_bottom_centers:
                    x,y = center[0], center[1]
                    track_id = players_labels[track_id_index]
                    self.player_trackers, x, y = self.kalman_filter_obj.smoothed_player_goalkeeper_referee_position_using_kalman_filter \
                                                        (self.player_trackers, track_id, x, y)
                    
                    x, y = int(x), int(y)
                    smoothed_x_y_list.append([x,y])
                    cv2.circle(annotated_frame, (x, y), 21, (255, 255, 0), 1)
                    # If id is greater than 9 then text will be moved left direction to set in center
                    text_offset = -23 if track_id > 9 else -11  
                    cv2.putText(annotated_frame, str(track_id),
                                        (x + text_offset,y + 13),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    track_id_index += 1
                    
        return annotated_frame, smoothed_x_y_list, players_labels, ball_bbox
    

    def draw_football_center(self, pitch, transformer, bbox, color=(255, 255, 0), radius=10, thickness=-1):
        try:
            x1, y1, x2, y2 = bbox
            # Compute center coordinates
            cx = x1 + (x2 - x1) * 0.5
            cy = y1 + (y2 - y1) * 0.5
            cx_cy = transformer.transform_points(np.array([list([cx,cy])]))
            cx = int(cx_cy[0][0] * 0.1) + 50
            cy = int(cx_cy[0][1] * 0.1) + 50
            # Draw circle at the center
            cv2.circle(pitch, (cx, cy), radius, color, thickness)
            return pitch
        except Exception as e:
            print(f"Error drawing football: {e}")
            return pitch

    def draw_football_hitmap(self,players_referees_goalkeepers_df,frame_count):
        a=4

    def draw_player_position_on_2Dpitch(
            self, transformer, smoothed_x_y_list, track_id_list, team_classifier, ball_bbox, frame_count, 
            players_referees_goalkeepers_df,
            config=SoccerPitchConfiguration(), face_color: sv.Color = sv.Color.RED, edge_color: sv.Color = sv.Color.BLACK,
            radius: int = 10, thickness: int = 2, padding: int = 50, scale: float = 0.1) -> np.ndarray:
        
        smoothed_x_y_transformed = transformer.transform_points(np.array(smoothed_x_y_list))
        # 1) initialise pitch canvas
        # if pitch is None:
        #     pitch = draw_pitch(self.CONFIG)
        print(" ------------------------------- : ",len(ball_bbox), ball_bbox)
        if all(len(inner) == 0 for inner in ball_bbox):
            pass
        else:
            self.pitch = self.draw_football_center(self.pitch, transformer, ball_bbox[0])
        track_id_index = 0
        pitch = self.pitch.copy()
        for foot_pitch in smoothed_x_y_transformed:
            # 5) pitch cm → canvas px
            px_x = int(foot_pitch[0] * scale) + padding
            px_y = int(foot_pitch[1] * scale) + padding
            track_id__ = track_id_list[track_id_index]
            if team_classifier[track_id__] ==0:
                cv2.circle(pitch, (px_x, px_y), 25, (0, 0, 255), 3)
                text_offset = -23 if track_id__> 9 else -11  # If id is greater than 9 then text will be moved left direction to set in center
                cv2.putText(pitch, str(track_id__),
                            (px_x + text_offset, px_y + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            elif team_classifier[track_id__] ==1:
                cv2.circle(pitch, (px_x, px_y), 25, (0, 255, 0), 3)
                text_offset = -23 if track_id__ > 9 else -11  # If id is greater than 9 then text will be moved left direction to set in center
                cv2.putText(pitch, str(track_id__),
                            (px_x + text_offset, px_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            elif team_classifier[track_id__] ==2:
                cv2.circle(pitch, (px_x, px_y), 25, (255, 0, 0), 3)
                text_offset = -23 if track_id__ > 9 else -11  # If id is greater than 9 then text will be moved left direction to set in center
                cv2.putText(pitch, str(track_id__),
                            (px_x + text_offset, px_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            track_id_index += 1
        return pitch
        
    def draw_32_football_field_keypoints(self, annotated_frame, frame_count, field_keypoints_df, players_referees_goalkeepers_df):
        # ---- Extract field_keypoints from df for this frame ----
        if frame_count in field_keypoints_df["Frame_Num"].values:
            row = field_keypoints_df.loc[field_keypoints_df["Frame_Num"] == frame_count].iloc[0]
            # frame_reference_points = row["frame_reference_points"]  # this is already a dict
            cleaned_keypoints_dict = row["cleaned_keypoints"]
            # cleaned_keypoints_dict = ast.literal_eval(cleaned_keypoints_dict)
        else:
            cleaned_keypoints_dict = {}  # fallback if no match found
        
        if frame_count in players_referees_goalkeepers_df["Frame_Num"].values:
            row = players_referees_goalkeepers_df.loc[players_referees_goalkeepers_df["Frame_Num"] == frame_count].iloc[0]
            # frame_reference_points = row["frame_reference_points"]  # this is already a dict
            team_classifier = row["team_classifier"]
            team_classifier = ast.literal_eval(team_classifier)
            # team_classifier = team_classifier.values()
            # team_classifier = list(team_classifier)
            # team_classifier = np.fromstring(team_classifier.strip("[]"), dtype=int, sep=" ").tolist()
        else:
            team_classifier = {}  # fallback if no match found
            
        # Convert string to dictionary
        # frame_reference_points = ast.literal_eval(frame_reference_points)
        cleaned_keypoints_value = list(cleaned_keypoints_dict.values())
        frame_reference_points = np.array(cleaned_keypoints_value)
        frame_reference_length = len(frame_reference_points)
        print("len(frame_reference_points) : ",len(frame_reference_points))
        if frame_reference_length==0:
            return annotated_frame, frame_reference_length, None, None, None
        # reference keypoints for frame
        frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
        
        # Create mask of detected class IDs
        mask = np.zeros(32, dtype=bool)  # assumes max 32 reference classes
        cleaned_sorted_dict_k = list(cleaned_keypoints_dict.keys())
        mask[cleaned_sorted_dict_k] = True
        pitch_reference_points = np.array(self.CONFIG.vertices)[mask]
        
        # homography transformation
        transformer = ViewTransformer(source=pitch_reference_points , target=frame_reference_points)

        # transform all pitch vertices into frame space
        pitch_all_points = np.array(self.CONFIG.vertices)
        frame_all_points = transformer.transform_points(points=pitch_all_points)
        frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

        # # annotate edges and vertices
        # annotated_frame = self.edge_annotator.annotate(
        #     scene=annotated_frame,
        #     key_points=frame_all_key_points)

        # annotated_frame = self.vertex_annotator_2.annotate(
        #     scene=annotated_frame,
        #     key_points=frame_all_key_points)

        # # annotated_frame = self.vertex_annotator.annotate(
        #     scene=annotated_frame,
        #     key_points=frame_reference_key_points)
        return  annotated_frame, frame_reference_length, frame_reference_points, pitch_reference_points, team_classifier
    
    def get_video_writer(self, video_name, frame_size=(1480, 1080), fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = f"/media/gflml/Volume E/yolo/Soccer/output/{video_name}.mp4"
        return cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    def get_merged_image_vertically(self, annotated_frame,pitch_frame):
        # Resize images to 640x640
        annotated_frame = cv2.resize(annotated_frame, (740, 1080))
        pitch_frame = cv2.resize(pitch_frame, (740, 1080))
        # Stack vertically
        merged_image = cv2.vconcat([annotated_frame,pitch_frame])
        # Resize the merged image to 1280x720
        merged_image_resized = cv2.resize(merged_image, (1480, 1080))
        return merged_image_resized


    def main_func_to_draw_32_football_field_keypoints(self, input_video_path, field_keypoints_df, players_referees_goalkeepers_df):
        cap = cv2.VideoCapture(input_video_path)
        frame_count = 0  # Counter for frames    
        video_writer = self.get_video_writer("output_from_images",(1480, 1080))
        while cap.isOpened():
            frame_count+=1
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends
            if frame_count%10 == 0:
                print("----------------- frame_count ---------------- : ",frame_count)
            # if(frame_count<240):
            #     continue
            if(frame_count>5041):
                break

            # ---- Call your function with current frame & keypoints ----     
            annotated_frame, frame_reference_length, frame_reference_points, pitch_reference_points, team_classifier = \
                self.draw_32_football_field_keypoints(frame,frame_count, field_keypoints_df, players_referees_goalkeepers_df) # Function Call
            if frame_reference_length == 0:
                break 
            annotated_frame, smoothed_x_y_list, players_labels, ball_bbox = self.draw_bbox_with_label(annotated_frame,
                                                                frame_count, players_referees_goalkeepers_df) # Function Call
            frame_reference_points = np.array([
                                [123, 370.5],
                                [1084.5, 330],
                                [1031, 586],
                                [957, 948],
                                [1353, 514]])
            pitch_reference_points = np.array([
                                [2015, 1450],
                                [6000, 0],
                                [6000, 4415],
                                [6000, 7000],
                                [6915, 3500]])
            
            # frame_reference_points = np.array([[        178,       369.5],
            #                 [         94,         393],
            #                 [      657.5,         351],
            #                 [      643.5,         391],
            #                 [        623,       447.5],
            #                 [       1141,       370.5],
            #                 [       1220,         396],
            #                 [      469.5,         418],
            #                 [      798.5,       414.5]])
            # pitch_reference_points = np.array([[       2015,        1450],
            #                 [       2015,        2584],
            #                 [       6000,           0],
            #                 [       6000,        2585],
            #                 [       6000,        4415],
            #                 [       9985,        1450],
            #                 [       9985,        2584],
            #                 [       5085,        3500],
            #                 [       6915,        3500]])
            
            transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
            pitch_frame = self.draw_player_position_on_2Dpitch(transformer, smoothed_x_y_list, players_labels, team_classifier, 
                                            ball_bbox, frame_count, players_referees_goalkeepers_df) # Function Call
            # annotated_frame = cv2.resize(annotated_frame,(1300,800))
            cv2.putText(annotated_frame, str(frame_count),(30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            output_frame = self.get_merged_image_vertically(annotated_frame, pitch_frame)
            video_writer.write(output_frame)
        video_writer.release()
        print("✅ Output video saved.")        
                    
                    
                    