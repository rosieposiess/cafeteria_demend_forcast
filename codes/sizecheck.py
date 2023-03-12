import cv2

model = 'yolov3.weights'
config = 'yolov3.cfg'

# 모델 로드
net = cv2.dnn.readNet(model, config)

# 모델의 입력 레이어 이름 확인
layer_names = net.getLayerNames()
input_layer = layer_names[0]

# 입력 레이어 객체 가져오기
input_layer = net.getLayer(input_layer)

# 입력 레이어의 출력 형태 확인
input_shape = input_layer.outputShape

# 입력 이미지의 너비와 높이 확인
image_height = input_shape[2]
image_width = input_shape[3]


