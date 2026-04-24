# ⚽ Football Field, Player & Ball Tracking System

An intelligent **football analytics system** built using **Computer Vision and Deep Learning** techniques.  
This project combines object detection, tracking, and geometric transformation to analyze football matches and track players smoothly on the field.

---

##  screenshot
![]([output/output img.png](https://github.com/sohag-mbstu-ict/Sports-Tracking-and-Tactical-Analytics/blob/main/output/output%20img.png))

##  Video Demo
https://raw.githubusercontent.com/sohag-mbstu-ict/Sports-Tracking-and-Tactical-Analytics/main/output/output.mp4

## Features

-  **Football Field Keypoint Detection**
  - Detects field landmarks using YOLO-based model

-  **2D Pitch Transformation**
  - Applies homography to map players onto a top-down pitch view

-  **Player, Referee & Goalkeeper Detection**
  - Multi-class detection using YOLO models

-  **Multi-Object Tracking**
  - Uses **ByteTrack** for consistent player tracking across frames

-  **Motion Smoothing**
  - Applies **Kalman Filter** to reduce tracking noise and improve stability

-  **Team Classification**
  - Clusters players into teams (e.g., red & green) using **K-Means**

---

##  Tech Stack

- Python
- OpenCV
- YOLO (Object Detection)
- ByteTrack (Tracking)
- Kalman Filter (Smoothing)
- NumPy / Scikit-learn (Clustering)

---

##  Pipeline Overview

1. Detect field keypoints  
2. Apply homography transformation  
3. Detect players, referees, and ball  
4. Track objects across frames using ByteTrack  
5. Smooth trajectories with Kalman Filter  
6. Cluster players into teams  

---

