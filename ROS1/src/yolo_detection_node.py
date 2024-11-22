#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO, SAM

class CameraDisplayNode:
    def __init__(self):
        rospy.init_node('camera_display_node', anonymous=True)

        # Load YOLO and SAM models
        self.model = YOLO("/home/swapnil/catkin_ws/src/yolo_detection/best.pt")
        self.sam_model = SAM("/home/swapnil/catkin_ws/src/yolo_detection/sam2_b.pt")
        
        # Create a CvBridge instance
        self.bridge = CvBridge()
        
        # Subscribe to RGB and depth topics
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)

        # Initialize frame storage
        self.frame = None
        self.depth_frame = None

    def rgb_callback(self, msg):
        try:
            # Convert RGB message to OpenCV format
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rospy.loginfo("Received RGB frame")

            # Run YOLO and SAM only on RGB data
            self.process_segmentation()

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def depth_callback(self, msg):
        try:
            # Convert the depth message from 16UC1 to meters
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0  # Convert mm to meters
            rospy.loginfo("Received Depth frame")

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_segmentation(self):
        if self.frame is None:
            return
        
        # Run YOLO detection on the RGB frame
        results = self.model(self.frame)
        
        for result in results:
            boxes = result.boxes.xyxy.tolist() if result.boxes is not None else []
            
            if boxes:
                rospy.loginfo(f"YOLO bounding boxes: {boxes}")

                # Run SAM on the bounding boxes
                sam_results = self.sam_model(result.orig_img, bboxes=boxes, verbose=False)

                # Access the masks from the SAM results
                if sam_results and sam_results[0].masks is not None:
                    for mask_data, box in zip(sam_results[0].masks.data, boxes):
                        mask_data = mask_data.cpu().numpy().astype(np.uint8) * 255
                        colored_mask = np.zeros_like(self.frame)
                        colored_mask[mask_data > 0] = (0, 255, 0)

                        # Blend mask for visualization
                        self.frame = cv2.addWeighted(self.frame, 1, colored_mask, 0.5, 0)

                        # Optionally overlay depth information
                        if self.depth_frame is not None:
                            x1, y1, x2, y2 = map(int, box)
                            depth_region = self.depth_frame[y1:y2, x1:x2]
                            mean_depth = np.mean(depth_region[np.isfinite(depth_region)])
                            rospy.loginfo(f"Mean depth for box {box}: {mean_depth:.2f} meters")
                            label = f"Depth: {mean_depth:.2f}m"
                            cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    rospy.logwarn("SAM returned no masks or unexpected format.")
            else:
                rospy.logwarn("No valid bounding boxes detected for SAM processing.")

        # Display the frame with YOLO detections and SAM segmentation masks
        cv2.imshow("YOLO + SAM Segmentation + Depth", self.frame)
        cv2.waitKey(1)

def main():
    node = CameraDisplayNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
