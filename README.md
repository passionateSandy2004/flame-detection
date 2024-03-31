# Fire Detection using YOLOv5

This script utilizes YOLOv5 for detecting fire in a video stream. It processes each frame of the video, applies YOLOv5 object detection to identify potential fire instances, and displays an alert when fire is detected.

## Requirements
- Python 3.x
- OpenCV
- cvzone
- YOLOv5

## Usage
1. Ensure all dependencies are installed.
2. Replace `'video.mp4'` with the path to your input video.
3. Run the script.

## Parameters
- `best.pt`: Pre-trained YOLOv5 model for object detection.
- `classnames`: List of classes recognized by the YOLO model, here only containing 'fire'.
- `lower` and `upper`: Lower and upper HSV (Hue, Saturation, Value) values for fire detection.

## Description
1. The script initializes the video capture from `'video.mp4'`.
2. It loads the YOLOv5 model for object detection.
3. It sets up a loop to process each frame of the video.
4. Fire detection is performed using a combination of HSV color filtering and YOLOv5 object detection.
5. When fire is detected, an alert is displayed on the frame.
6. The processed frame is displayed along with the fire detection output.
7. Press 'q' to quit the application.

## Additional Notes
- Adjust the `lower` and `upper` HSV values for better fire detection in different environments.
- Ensure proper lighting conditions for accurate detection.
