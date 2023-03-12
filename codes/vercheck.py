import cv2

# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get layer names and output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Print network information
for layer in layer_names:
    layer_info = net.getLayer(layer)
    print("Layer name:", layer_info.name)
    print("Layer type:", layer_info.type)

# Print output layers
print("Output layers:", output_layers)
