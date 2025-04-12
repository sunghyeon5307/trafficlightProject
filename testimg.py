import cv2
from ultralytics import YOLO

model = YOLO('/Users/bagseonghyeon/Documents/traffic_light/trafficlight_model/best.pt') 
model2 = YOLO('/Users/bagseonghyeon/Documents/traffic_light/lightcolor_model/best.pt')

img_path = "/Users/bagseonghyeon/Documents/traffic_light/testimg/red.jpeg"
image = cv2.imread(img_path)

result1 = model(image)

for r in result1:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]  # numpy 이미지 자르기

        # 2단계: 잘라낸 영역에서 색상 탐지
        result2 = model2(cropped)

        # 결과 시각화 (box + label 포함)
        result_img = result2[0].plot()
        cv2.imshow("Detected Traffic Light Color", result_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
