import cv2
import serial
import time
from ultralytics import YOLO

ser = serial.Serial('/dev/cu.usbmodem11201', 9600)  
time.sleep(2)

model_tl = YOLO('/Users/bagseonghyeon/Documents/traffic_light/trafficlight_model/best.pt') 
model_color = YOLO('/Users/bagseonghyeon/Documents/traffic_light/lightcolor_model/best.pt')

cap = cv2.VideoCapture('/Users/bagseonghyeon/Documents/traffic_light/testimg/trafficlight.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_tl = model_tl(frame)

    for r in result_tl:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]

            result_color = model_color(cropped)

            if result_color[0].boxes and result_color[0].boxes.cls.numel() > 0:
                class_id = int(result_color[0].boxes.cls[0])
                label = model_color.names[class_id]

                color = (0, 255, 0) if label == "green_light" else (0, 255, 255) if label == "yellow_light" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if label == "red_light":
                    ser.write(b"0\n")
                elif label == "green_light":
                    ser.write(b"2\n")
                elif label == "yellow_light":
                    ser.write(b"1\n")

                print("Detected:", label)
            else:
                print("No color detected in cropped box")


    cv2.imshow("Traffic Light Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
