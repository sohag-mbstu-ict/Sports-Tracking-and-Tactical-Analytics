import os
import cv2
from typing import Optional, List
import numpy as np
import supervision as sv
from tqdm import tqdm
from inference import get_model
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from ultralytics import YOLO
from pre_and_post_processing import GetInfo, ImageAndVideoProcessing

"""Milestone 1.
◾Football Field Keypoint Detection 
◾2D Pitch Transformation 
◾Player, Referee & Goalkeeper Detection 
◾Continuous Player Tracking
"""
class FootballAIPipeline:
    """
    Class to handle football video analysis:
    - Player, ball, goalkeeper, referee detection
    - Team classification
    - Field key points detection
    - Projection of objects on pitch
    - Visualization of annotated frames
    """

    def __init__(self):

        # Load models
        self.keypoints_model = YOLO("/home/gflmltpc/YOLO_model/trained_model/field_keypoints_detector.pt")
        self.detection_model = YOLO("/home/gflmltpc/YOLO_model/trained_model/Player_Referee_Goalkeeper_Detection.pt")
        # self.detection_model = YOLO("/home/gflmltpc/YOLO_model/yolo11x.pt")
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3
        # Soccer pitch configuration
        self.CONFIG = SoccerPitchConfiguration()

        # Annotators
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex('#FF1493'), radius=8)
        self.edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex('#00BFFF'), thickness=2, edges=self.CONFIG.edges)
        self.vertex_annotator_2 = sv.VertexAnnotator(color=sv.Color.from_hex('#00BFFF'), radius=8)
        
    def categorize_detections_using_ID(self,detections):
        
        ball_detections = detections[detections.class_id == self.BALL_ID]
        goalkeeper_detections = detections[detections.class_id == self.GOALKEEPER_ID]
        player_detections = detections[detections.class_id == self.PLAYER_ID]
        referee_detections = detections[detections.class_id == self.REFEREE_ID]
        player_goalkeeper_ball_detections = sv.Detections.merge([
                                     player_detections, goalkeeper_detections,ball_detections])
        return player_goalkeeper_ball_detections

    def detect_and_annotate_frame(self, frame,tracker):
        """Detect all entities in the frame, classify teams, and annotate."""

        # result = self.PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        results = self.detection_model(frame,imgsz = 1280)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        ball_detections = all_detections[all_detections.class_id == self.BALL_ID]
        player_goalkeeper_ball_detections = self.categorize_detections_using_ID(all_detections) # categorize ball, goalkeepers, player, referee
        player_goalkeeper_detections = player_goalkeeper_ball_detections[player_goalkeeper_ball_detections.class_id != self.BALL_ID]
        player_goalkeeper_detections = tracker.update_with_detections(detections=player_goalkeeper_detections)
        labels = [
            f"#{tracker_id}"
            for tracker_id
            in player_goalkeeper_detections.tracker_id]
                
        annotated_frame = frame.copy()
        annotated_frame = self.ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=player_goalkeeper_detections,
            labels=labels)
        return annotated_frame, player_goalkeeper_detections,ball_detections
    
    def draw_football_field_keypoints(self,key_points,annotated_frame):
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        frame_reference_key_points = sv.KeyPoints(
            xy=frame_reference_points[np.newaxis, ...])
        pitch_reference_points = np.array(self.CONFIG.vertices)[filter]
        transformer = ViewTransformer(
            source=pitch_reference_points,
            target=frame_reference_points)
        pitch_all_points = np.array(self.CONFIG.vertices)
        frame_all_points = transformer.transform_points(points=pitch_all_points)
        frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
        annotated_frame = self.edge_annotator.annotate(
            scene=annotated_frame,
            key_points=frame_all_key_points)
        annotated_frame = self.vertex_annotator_2.annotate(
            scene=annotated_frame,
            key_points=frame_all_key_points)
        annotated_frame = self.vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=frame_reference_key_points)
        return annotated_frame
            
    def detect_field_keypoints_using_custom_trained_model(self, frame, annotated_frame, conf_threshold=0.5):
        result = self.keypoints_model(frame,imgsz=1280, verbose=False)[0]      # ultralytics.engine.results.Results
        #  3. Wrap keypoints with supervision
        key_points = sv.KeyPoints.from_ultralytics(result)
        filter_mask = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter_mask]
        pitch_reference_points = np.array(self.CONFIG.vertices)[filter_mask]
        transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
        annotated_frame = self.draw_football_field_keypoints(key_points,annotated_frame) # draw filed keypoints and line
        return transformer, annotated_frame
    
    def draw_points_on_pitch(self,track_id, player_goalkeeper_xy,ball_detections_xy,
        config: SoccerPitchConfiguration,
        face_color: sv.Color = sv.Color.RED,
        edge_color: sv.Color = sv.Color.BLACK,
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
        ) -> np.ndarray:
        
        if pitch is None:
            pitch = draw_pitch(
                config=config,
                padding=padding,
                scale=scale)
        if(len(ball_detections_xy)>0):    
            for point in ball_detections_xy:
                scaled_point = ( int(point[0] * scale) + padding, int(point[1] * scale) + padding)
                cv2.circle(img=pitch, center=scaled_point, radius=17,color=(0,0,0),thickness=-1)
            
        track_id_index = 0
        for point in player_goalkeeper_xy:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding)
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=25,
                color=(255,255,0),
                thickness=3)
            if(track_id[track_id_index]>9):
                # # Add text "13" inside the circle (centered)
                cv2.putText(
                    img=pitch,
                    text=str(track_id[track_id_index]),
                    org=(scaled_point[0]-23, scaled_point[1]+13),  # adjust offset for centering
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),  # white text
                    thickness=2,
                    lineType=cv2.LINE_AA)
            else:
                # # Add text "1" inside the circle (centered)
                cv2.putText(
                    img=pitch,
                    text=str(track_id[track_id_index]),
                    org=(scaled_point[0]-11, scaled_point[1]+13),  # adjust offset for centering
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),  # white text
                    thickness=2,
                    lineType=cv2.LINE_AA)
            track_id_index += 1
        
        return pitch
        
    def project_on_pitch(self, frame, transformer, player_goalkeeper_detections,ball_detections):
        """Project ball, players, referees onto pitch."""
        pitch_frame = draw_pitch(self.CONFIG)
        tracking_id = player_goalkeeper_detections.tracker_id
        player_goalkeeper_xy = transformer.transform_points(player_goalkeeper_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        ball_detections_xy = transformer.transform_points(ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        # Draw objects on pitch
        pitch_frame = self.draw_points_on_pitch(tracking_id, player_goalkeeper_xy, ball_detections_xy, self.CONFIG,
                    face_color=sv.Color.from_hex('00BFFF'), edge_color=sv.Color.BLACK, 
                    radius=16, pitch=pitch_frame)
        return pitch_frame
    
    def get_merged_image_vertically(self, annotated_frame,pitch_frame):
        # Resize images to 640x640
        annotated_frame = cv2.resize(annotated_frame, (740, 1080))
        pitch_frame = cv2.resize(pitch_frame, (740, 1080))
        # Stack vertically
        merged_image = cv2.vconcat([annotated_frame,pitch_frame])
        # Resize the merged image to 1280x720
        merged_image_resized = cv2.resize(merged_image, (1480, 1080))
        return merged_image_resized

    def run_pipeline(self, save_image_path="annotated_pitch.jpg"):
        """Run the full pipeline on one frame and save the result."""
        infer_obj = GetInfo()
        video_writer = infer_obj.get_video_writer("output_from_images",(1480,1080))
        # video_writer = infer_obj.get_video_writer("output_from_images",(1300,800))
        
        input_video_path = "/home/gflmltpc/YOLO_model/football-AI/videos/0bfacc_0.mp4"
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0  # Counter for frames 
        tracker = sv.ByteTrack()
        tracker.reset()       
        while cap.isOpened():
            frame_count+=1
            if frame_count%10 == 0:
                print("----------------- frame_count ---------------- : ",frame_count)
            if(frame_count>290):
                break
            ret, frame = cap.read()
            if not ret:
                break  # Stop if video ends
            
            # 3. Annotate frame with detections
            annotated_frame, player_goalkeeper_detections,ball_detections = self.detect_and_annotate_frame(frame,tracker)
            # 4. Detect field keypoints and get transformer
            transformer, annotated_frame = self.detect_field_keypoints_using_custom_trained_model(frame,annotated_frame)
            # 5. Project everything on pitch
            pitch_frame = self.project_on_pitch(frame, transformer, player_goalkeeper_detections,ball_detections)
            output_image = self.get_merged_image_vertically(annotated_frame,pitch_frame)
            video_writer.write(output_image)
            # pitch_frame = cv2.resize(pitch_frame,(1300,800))
            # video_writer.write(pitch_frame)
        video_writer.release()
        print("✅ Output video saved.")

# ================= Usage =================
if __name__ == "__main__":

    pipeline = FootballAIPipeline()
    pipeline.run_pipeline()
