import numpy as np
import supervision as sv
from tqdm import tqdm
import cv2
from sports.common.team import TeamClassifier


class team_classification:
    def __init__(self):
        self.team_classifier_obj = TeamClassifier(device="cuda")
        self.ellipse_annotator = sv.EllipseAnnotator(
                color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER)
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1)
        
    def team_classifier(self,frame, players_bbox_with_id, referees_detections, goalkeepers_detections, ball_detections):
        players_detections = players_bbox_with_id.values()
        players_detections_keys = players_bbox_with_id.keys()
        players_detections = np.array(list(players_detections))
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections]
        players_detections = self.team_classifier_obj.predict(players_crops)
        # mapping team ID (team ID 0 or 1) to the label name
        players_team_ID = dict(zip(players_detections_keys, players_detections.tolist()))
        # goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        #     players_detections, goalkeepers_detections)
        # referees_detections.class_id -= 1
        # all_detections = sv.Detections.merge([
        #     players_detections, goalkeepers_detections, referees_detections])
        
        # referees_detections.class_id -= 1
        # all_detections = players_detections
        # # all_detections = sv.Detections.merge([
        # #     players_detections,goalkeepers_detections])#, referees_detections])
        

        # labels = [
        #     f"#{tracker_id}"
        #     for tracker_id
        #     in all_detections.tracker_id
        # ]

        # all_detections.class_id = all_detections.class_id.astype(int)
        # class_id_list_for_team_classify = all_detections.class_id

        annotated_frame = frame.copy()
        # annotated_frame = self.ellipse_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=all_detections)
        # annotated_frame = self.label_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=all_detections,
        #     labels=labels)
        # annotated_frame = self.triangle_annotator.annotate(
        #     scene=annotated_frame,
        #     detections=ball_detections)
        # a=4
        return annotated_frame, players_team_ID
        
        
#   File "<unknown>", line 1
#     [[1514.3,558.62,1545.8,591.38],[,1513.9,557.82,1547.1,593.73]]