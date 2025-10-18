import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import supervision as sv
from ultralytics import YOLO
from typing import Optional
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration


class smoothed_position_using_kalman_filter:
    def __init__(self):
        pass
                
    # # Kalman-filter factory
    def create_kf_2d(self,dt=1.0, q=0.1):
        """
        2-D constant-velocity Kalman filter.
        State: [x, vx, y, vy]
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0., 0., 0., 0.])          # initial state
        kf.F = np.array([[1, dt, 0,  0],
                        [0,  1, 0,  0],
                        [0,  0, 1, dt],
                        [0,  0, 0,  1]])
        kf.H = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])
        kf.R *= 5.0                                 # observation noise
        kf.Q *= q                                   # process noise
        return kf

    def smoothed_player_goalkeeper_referee_position_using_kalman_filter(self, trackers, track_id, x, y):
        # reuse or create tracker dict (keep across frames if you make it an instance member)
        foot_px = np.array([x, y], dtype=np.float32)
        # ---------- Kalman smoothing ----------
        if track_id not in trackers:
            kf = self.create_kf_2d(dt=1/30, q=0.01)
            kf.x = np.array([foot_px[0], 0., foot_px[1], 0.])
            trackers[track_id] = kf
        else:
            kf = trackers[track_id]
        kf.predict()
        kf.update(foot_px)
        smoothed_x, smoothed_y = kf.x[0], kf.x[2]
        # if track_id in (2, 9, 13):
        #     print(f"   Player {track_id}: x={x}, y={y}")
        #     print("smoothed_x, smoothed_y :", smoothed_x, smoothed_y)
        return trackers, smoothed_x, smoothed_y






    # def smoothed_player_goalkeeper_referee_position_using_kalman_filter(self, df):
    #     # reuse or create tracker dict (keep across frames if you make it an instance member)
    #     if not hasattr(self, 'trackers'):
    #         self.trackers = {}

    #     # add empty column first
    #     df = df.copy()
    #     df["updated_player_bbox"] = None  

    #     # loop through rows
    #     for i, row in enumerate(df.itertuples(index=False)):
    #         players = row.Players   # dict
    #         smoothed_dict = {}

    #         for track_id, coords in players.items():
    #             x, y = coords
    #             foot_px = np.array([x, y], dtype=np.float32)

    #             # ---------- Kalman smoothing ----------
    #             if track_id not in self.trackers:
    #                 kf = self.create_kf_2d(dt=1/30, q=0.01)
    #                 kf.x = np.array([foot_px[0], 0., foot_px[1], 0.])
    #                 self.trackers[track_id] = kf
    #             else:
    #                 kf = self.trackers[track_id]

    #             kf.predict()
    #             kf.update(foot_px)
    #             smoothed_x, smoothed_y = kf.x[0], kf.x[2]

    #             if track_id in (8, 11, 13):
    #                 print(f"   Player {track_id}: x={x}, y={y}")
    #                 print("smoothed_x, smoothed_y :", smoothed_x, smoothed_y)

    #             smoothed_dict[track_id] = [int(smoothed_x), int(smoothed_y)]

    #         # ✅ assign dict directly to new column
    #         df.at[i+1, "updated_player_bbox"] = smoothed_dict

    #         print("--------------------------------------------------")

    #     return df


