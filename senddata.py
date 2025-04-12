import cv2
import serial
import time
from ultralytics import YOLO

ser = serial.Serial('/dev/cu.usbmodem11201', 9600)  
time.sleep(2)

model = YOLO('/Users/bagseonghyeon/Documents/traffic_light/trafficlight_model/best.pt') 
model2 = YOLO('/Users/bagseonghyeon/Documents/traffic_light/lightcolor_model/best.pt')

img_path = "/Users/bagseonghyeon/Documents/traffic_light/testimg/test2.jpeg"
image = cv2.imread(img_path)

result1 = model(image)

for r in result1:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]

        result2 = model2(cropped)
        result_img = result2[0].plot()

        class_id = int(result2[0].boxes.cls[0])
        label = model2.names[class_id]

        if label == "red_light":
            ser.write(b"0\n")
        elif label == "green_light":
            ser.write(b"2\n")
        elif label == "yellow_light":
            ser.write(b"1\n")

        cv2.imshow("img", result_img)
        print("Detected:", label)
        cv2.waitKey(0)

cv2.destroyAllWindows()
ser.close()
