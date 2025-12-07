import cv2
import numpy as np
import time

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MIN_CONTOUR_AREA = 1500

WARNING_DIST = 150
DANGER_DIST = 40

# RED color HSV ranges
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])

RED_LOWER_2 = np.array([160, 120, 70])
RED_UPPER_2 = np.array([179, 255, 255])

OBJ_CENTER = (int(FRAME_WIDTH * 0.75), int(FRAME_HEIGHT * 0.5))
OBJ_RADIUS = 60

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def get_fingertip(cnt):
    return tuple(cnt[cnt[:, :, 1].argmin()][0])


cap = cv2.VideoCapture(0)
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color segmentation for RED
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    state = "SAFE"
    fingertip = None
    dist = None

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            fingertip = get_fingertip(cnt)

            dx = fingertip[0] - OBJ_CENTER[0]
            dy = fingertip[1] - OBJ_CENTER[1]
            dist = ((dx * dx + dy * dy) ** 0.5) - OBJ_RADIUS

            if dist <= DANGER_DIST:
                state = "DANGER"
            elif dist <= WARNING_DIST:
                state = "WARNING"

            cv2.circle(frame, fingertip, 8, (0, 255, 255), -1)

    cv2.circle(frame, OBJ_CENTER, OBJ_RADIUS, (200, 200, 200), 2)
    cv2.putText(frame, f"State: {state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if state == "DANGER":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.putText(frame, "DANGER DANGER",
                    (150, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 5)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1 / (now - prev_time))
    prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Red Mask", mask)
    cv2.imshow("Hand POC - Red Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
