import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

pixels_per_cm = 25.21  # Calibrated value: pixels per centimeter

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse at: {x}, {y}")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

model = YOLO("best.pt")  # Load your segmentation model
names = model.names

cap = cv2.VideoCapture(0)
count = 0

def clip_line_to_contour(p1, p2, contour):
    mask = np.zeros((500, 1020), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    line_img = np.zeros_like(mask)
    cv2.line(line_img, p1, p2, 255, 1)
    intersection = cv2.bitwise_and(mask, line_img)
    coords = cv2.findNonZero(intersection)
    if coords is not None and len(coords) >= 2:
        a = tuple(coords[0][0])
        b = tuple(coords[-1][0])
        return a, b
    return None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

        masks = results[0].masks
        if masks is not None:
            masks = masks.xy
            overlay = frame.copy()

            for box, track_id, class_id, mask in zip(boxes, track_ids, class_ids, masks):
                class_name = names[class_id]

                if mask.size > 0:
                    mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [mask], color=(0, 0, 255))
                    cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2)

                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

                        rect = cv2.minAreaRect(mask)
                        (center_x, center_y), (w, h), angle = rect
                        angle = np.deg2rad(angle)

                        width_cm = w / pixels_per_cm
                        height_cm = h / pixels_per_cm

                        dx = int((w / 2) * np.cos(angle))
                        dy = int((w / 2) * np.sin(angle))
                        x1_w, y1_w = int(cx - dx), int(cy - dy)
                        x2_w, y2_w = int(cx + dx), int(cy + dy)
                        clipped_w1, clipped_w2 = clip_line_to_contour((x1_w, y1_w), (x2_w, y2_w), mask)
                        if clipped_w1 and clipped_w2:
                            cv2.line(frame, clipped_w1, clipped_w2, (0, 255, 0), 2)
                            cvzone.putTextRect(frame, f"Width: {width_cm:.2f} cm", (clipped_w1[0], clipped_w1[1] - 30),
                                               scale=1, thickness=2, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=4)

                        dx_h = int((h / 2) * -np.sin(angle))
                        dy_h = int((h / 2) * np.cos(angle))
                        x1_h, y1_h = int(cx - dx_h), int(cy - dy_h)
                        x2_h, y2_h = int(cx + dx_h), int(cy + dy_h)
                        clipped_h1, clipped_h2 = clip_line_to_contour((x1_h, y1_h), (x2_h, y2_h), mask)
                        if clipped_h1 and clipped_h2:
                            cv2.line(frame, clipped_h1, clipped_h2, (255, 0, 0), 2)
                            cvzone.putTextRect(frame, f"Height: {height_cm:.2f} cm", (clipped_h1[0], clipped_h1[1] - 30),
                                               scale=1, thickness=2, colorT=(0, 0, 0), colorR=(255, 0, 255), offset=4)

                        # Show class name
                        cvzone.putTextRect(frame, f'{class_name.upper()}', (cx + 10, cy - 10),
                                           scale=1.2, thickness=2, colorT=(255, 255, 255),
                                           colorR=(0, 0, 255), offset=5, border=2)

            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
