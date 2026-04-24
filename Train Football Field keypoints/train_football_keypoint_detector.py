import argparse
from ultralytics import YOLO

def main():
    # Initialize model
    model = YOLO("yolo11x-pose.pt")
    # Train with augmentation disabled
    model.train(
        data="/home/gflmltpc/YOLO_model/Train_keypoints/football-field-detection-12/data.yaml",
        device="cuda",
        name="keypoint_football",
        exist_ok=True,
        batch=4,
        epochs=100 ,
        imgsz=640 ,
        mosaic=0.0 ,
        plots=True
        )

if __name__ == '__main__':
    main()
    
    