# car-detection-dashboard
This project is a computer vision pipeline designed to detect, track, and extract unique vehicle instances from video footage. Using YOLO Segmentation, the script isolates objects from their background and compiles them into a clean, organized gallery grid in real time.
I used the pretrained yolo26n-seg model to automatically mask the cars passing a certain point and displaying them in a side panel like screen using OpenCV. The masked picture is taken at the time when the car is visibly of best quality.
