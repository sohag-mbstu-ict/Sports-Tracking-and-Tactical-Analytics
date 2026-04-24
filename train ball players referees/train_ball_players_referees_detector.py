import argparse
from ultralytics import YOLO
# !yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml batch=6 epochs=50 imgsz=1280 plots=True

def main():
    # Initialize model
    model = YOLO("/home/gflmltpc/YOLO_model/yolo11x.pt")
    # Train with augmentation disabled
    model.train(
        data="/home/gflmltpc/YOLO_model/train_ball_players_referees_detector/football-players-detection-10/data.yaml",
        device="cuda",
        name="keypoint_football",
        exist_ok=True,
        batch=6,
        epochs=70 ,
        imgsz=1280 ,
        )

if __name__ == '__main__':
    main()
    
    