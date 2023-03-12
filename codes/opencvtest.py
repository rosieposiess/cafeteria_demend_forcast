import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')
import cv2


# HOG 기반 보행자 검출기 생성
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 이미지에서 보행자 검출
img = cv2.imread('IMG_1060.PNG')




# # 이미지 크기 조정
# scale_percent = 100# 50%로 축소
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# # 이미지 밝기 조정
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.equalizeHist(img)

# # 이미지 대비 조정
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img = clahe.apply(img)

# # 노이즈 제거
# img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)







found, _ = hog.detectMultiScale(img)

# 검출된 보행자 수 출력
print('Detected {} pedestrians in the input image.'.format(len(found)))

