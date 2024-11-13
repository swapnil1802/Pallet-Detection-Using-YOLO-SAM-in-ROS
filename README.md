# Pallet-Detection-Using-YOLO-SAM-in-ROS
Pallet detection in a warehouse setting using yolov8 finetuned on pallet dataset and segmented using SAM model in ROS1

The yolo V8 trained and finetuned model should be downloaded from: https://drive.google.com/file/d/12hk3OTOCA5ZUyz3EoQXK69h_czPzGZmZ/view?usp=sharing
The SAM model can be downloaded from:
The script for training the model can be downloaded from: https://colab.research.google.com/drive/1oQuPFNHG8WZIp5EXHr-2VpFERw113_IA?usp=sharing

The final workflow can detect pallets, floors, carts and flat surfaces and segment them as well along with detecting distance.
The workflow was tested in ROS1 with Realsense D435i camera module which has depth sensors as well.

For this project, various YOLO models were finetuned and tested. 
YOLO V11 detection model + SAM Model
YOLO V11 segmentation model
YOLO V8 Segmentation model
YOLO V8 + SAM Model.

Out of all this, YOLO V8 had better performance than others.

The images in the dataset were auto annotated using Grounding DINO and SAM model.
It was then uploaded to roboflow to verify the annotations and then augment the data.
LIghting conditions, orientations, salt & Pepper noise were added to augment the data.

The Training data was 70% of entire dataset
Testing data was 15%
Validation data was 15%
