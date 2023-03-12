import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')
import cv2
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 800, 600)

import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading image
img = cv2.imread("image.jpg")

# Getting dimensions of the image
height, width, channels = img.shape

# Preprocessing the input image
blob = cv2.dnn.blobFromImage(img, 1/255.0, (608,608), swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Run forward pass on the network
layer_outputs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-maximum suppression to get the final bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

# Display the output image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
