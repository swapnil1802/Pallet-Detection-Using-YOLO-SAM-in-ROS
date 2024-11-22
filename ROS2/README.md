# Pallet-Detection-Using-YOLO-SAM-in-ROS
Pallet detection in a warehouse setting using yolov8 finetuned on pallet dataset and segmented using SAM model in ROS1

The yolo V8 trained and finetuned model should be downloaded from: https://drive.google.com/file/d/12hk3OTOCA5ZUyz3EoQXK69h_czPzGZmZ/view?usp=sharing

The SAM model can be downloaded from:https://drive.google.com/file/d/1_3jgDOhT5fpXjwuN4GM2BH7xQaRILQV6/view?usp=drive_link

NOTE: After downloading these models, the paths specified in the yolo_detection _node.py need to be changed with the final paths of these files

The script for training the model can be downloaded from: https://colab.research.google.com/drive/1oQuPFNHG8WZIp5EXHr-2VpFERw113_IA?usp=sharing

Final segmentation of test images can be seen at: https://drive.google.com/drive/folders/1x9rt_8n9XX9neZvvFPeKhj-yt1FDudFn?usp=drive_link

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

Below are some samples of final segmented output:

![image](https://github.com/user-attachments/assets/6e53ed20-bd5c-4fa6-9b77-48c2e24a5037)
![image](https://github.com/user-attachments/assets/4bf91d0e-4ab0-4546-972a-1d79c6481088)
![image](https://github.com/user-attachments/assets/a7a8674e-5514-4d97-9ae1-7fd31036ccf4)

The training results of yolov8 model are:
![image](https://github.com/user-attachments/assets/e71a69fe-5128-4b82-a8f4-85d583fb3140)

The moderately good training results are a result of annotations of the dataset.
Improving the annotations will result in better results.

The model seems to be performing better in actual conditions.




