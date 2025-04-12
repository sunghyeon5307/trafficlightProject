import cv2
import serial
import time
from ultralytics import YOLO

ser = serial.Serial('/dev/cu.usbmodem1201', 9600)
time.sleep(2)

model = YOLO('/Users/bagseonghyeon/Documents/traffic_light/trafficlight_model/best.pt') 
model2 = YOLO('/Users/bagseonghyeon/Documents/traffic_light/lightcolor_model/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result1 = model(frame)

    for r in result1:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]

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

            print("Detected:", label)

            cv2.imshow("cam", result_img)

    cv2.imshow("cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
