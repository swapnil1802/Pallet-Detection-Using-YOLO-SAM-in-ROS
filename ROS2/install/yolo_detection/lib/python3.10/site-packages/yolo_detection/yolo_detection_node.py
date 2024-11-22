#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from rclpy.qos import QoSProfile, ReliabilityPolicy


class CameraDisplayNode(Node):
    def __init__(self):
        super().__init__('camera_display_node')

        #self.image_pub = self.create_publisher(Image, "processed_image", 10)

        print("Its running!!!")
        # Load YOLO and SAM models
        self.model = YOLO("/home/agrolab/Downloads/best.pt")
        self.sam_model = SAM("/home/agrolab/Downloads/sam2_b.pt")
        
        # Create a CvBridge instance
        self.bridge = CvBridge()

        # Create a QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10  # Queue size
        )
        
        # Subscribe to RGB and depth topics
        self.image_sub = self.create_subscription(
            Image,
            "/robot1/zed2i/left/image_rect_color",
            self.rgb_callback,
            qos_profile  # QoS profile depth
        )
        # self.depth_sub = self.create_subscription(
        #     Image,
        #     "/camera/depth/image_rect_raw",
        #     self.depth_callback,
        #     qos_profile  # QoS profile depth
        # )

        # Initialize frame storage
        self.frame = None
        self.depth_frame = None

    def rgb_callback(self, msg):
        try:
            # Convert RGB message to OpenCV format
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info("Received RGB frame")

            # Run YOLO and SAM only on RGB data
            self.process_segmentation()

        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    # def depth_callback(self, msg):
    #     try:
    #         # Convert the depth message from 16UC1 to meters
    #         self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0  # Convert mm to meters
    #         self.get_logger().info("Received Depth frame")

    #     except CvBridge.CvBridgeError as e:
    #         self.get_logger().error(f"CvBridge Error: {e}")

    def process_segmentation(self):
        if self.frame is None:
            print("Frame is None, cannot display image.")
            return
        
        # Run YOLO detection on the RGB frame
        results = self.model(self.frame)
        
        for result in results:
            boxes = result.boxes.xyxy.tolist() if result.boxes is not None else []
            
            if boxes:
                self.get_logger().info(f"YOLO bounding boxes: {boxes}")

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
                            self.get_logger().info(f"Mean depth for box {box}: {mean_depth:.2f} meters")
                            label = f"Depth: {mean_depth:.2f}m"
                            cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    #ros_msg = self.bridge.cv2_to_imgmsg(self.frame, "bgr8")
                    #self.image_pub.publish(ros_msg)
                else:
                    self.get_logger().warn("SAM returned no masks or unexpected format.")
            else:
                self.get_logger().warn("No valid bounding boxes detected for SAM processing.")

        # Display the frame with YOLO detections and SAM segmentation masks
        cv2.imwrite("YOLO_SAM_Segmentation_Depth.jpg", self.frame)
        # #cv2.waitKey(1)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):  # Optionally allow quitting with 'q'
        #     rclpy.shutdown()
        #     cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CameraDisplayNode()
    try:
        rclpy.spin(node)
    #except KeyboardInterrupt:
     #   print("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()