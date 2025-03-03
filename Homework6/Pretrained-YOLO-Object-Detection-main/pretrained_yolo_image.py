import os
import numpy as np
import cv2
from sort import Sort

tracker = Sort()

yolo_config_path = 'model/yolov3.cfg'
yolo_weights_path = 'model/yolov3.weights'

class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "minivan", "train", "truck", "boat",
                "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

# Load YOLO model
yolo_model = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
yolo_layers = yolo_model.getLayerNames()
yolo_output_layer = [yolo_layers[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

# Initialize tracking info for tracked car
tracked_enter_frame = None
tracked_exit_frame = None

# Process images
frame_number = 0
image_dir = "images"
for img_file in os.listdir(image_dir):
    frame_number += 1
    image_path = os.path.join(image_dir, img_file)
    img_to_detect = cv2.imread(image_path)
    img_height, img_width = img_to_detect.shape[:2]

    # Prepare blob for YOLO model
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(img_blob)
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    # Collect bounding boxes for NMS
    boxes = []
    confidences = []
    class_ids = []
    cars_count = 0
    minivans_count = 0

    for layer in obj_detection_layers:
        for detection in layer:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                box = detection[:4] * np.array([img_width, img_height, img_width, img_height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]

            # Append the filtered detection for SORT tracking
            detections.append([x, y, x + w, y + h, confidence])

            if class_id == class_labels.index("car"):
                cars_count += 1
            elif class_id == class_labels.index("minivan"):
                minivans_count += 1

    # Update tracker with filtered detections
    tracked_objects = tracker.update(np.array(detections))

    # Draw tracked objects with IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)

        if obj_id == 18:
            if tracked_enter_frame is None:
                tracked_enter_frame = frame_number
            tracked_exit_frame = frame_number
            color = (0, 0, 255)  # Red for tracked ID
        else:
            color = (0, 255, 0)  # Green for other IDs

        cv2.rectangle(img_to_detect, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_to_detect, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Print counts
    print(f"Frame {frame_number}: Cars = {cars_count}, Minivans = {minivans_count}")

    # Display or save the output
    cv2.imshow("Tracked Output", img_to_detect)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(f"Vehicle tracked entered at frame {tracked_enter_frame} and exited at frame {tracked_exit_frame}")
cv2.destroyAllWindows()
