import cv2
import sys

cam_id = sys.argv[1]  # Accepts camera ID from command line
cap = cv2.VideoCapture(int(cam_id) if cam_id.isdigit() else cam_id)

if not cap.isOpened():
    print("Cannot open camera:", cam_id)
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow(f'Camera {cam_id}', frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
