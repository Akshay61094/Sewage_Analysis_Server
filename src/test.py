import cv2

cap = cv2.VideoCapture(-1)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("", frame)
        cv2.waitKey(1)