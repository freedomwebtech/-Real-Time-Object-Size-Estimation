import tkinter as tk
from tkinter import simpledialog
import cv2
from PIL import Image, ImageTk
import math
import os

# === Globals ===
clicked_points = []
PIXEL_DISTANCE = None
PIXELS_PER_CM = None
CALIBRATION_FILE = "calibration.txt"

# === Load saved calibration (if any) ===
def load_calibration():
    global PIXELS_PER_CM
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            PIXELS_PER_CM = float(f.read().strip())
        print(f"✅ Loaded Calibration: {PIXELS_PER_CM:.2f} px/cm")
    else:
        print("⚠️ No saved calibration found. Please click two points to calibrate.")

# === Save calibration ===
def save_calibration(value):
    with open(CALIBRATION_FILE, "w") as f:
        f.write(str(value))
    print(f"💾 Saved Calibration: {value:.2f} px/cm")

# === Webcam Setup ===
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))
    if not ret:
        return

    frame_rgb = cv2.cvtColor(cv2.flip(frame, -1), cv2.COLOR_BGR2RGB)

    # Draw points
    for pt in clicked_points:
        cv2.circle(frame_rgb, pt, 5, (0, 0, 255), -1)

    if len(clicked_points) == 2:
        pt1, pt2 = clicked_points
        cv2.line(frame_rgb, pt1, pt2, (0, 255, 0), 2)
        pixel_distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

        if PIXELS_PER_CM:
            cm_distance = pixel_distance / PIXELS_PER_CM
            label = f"{pixel_distance:.0f}px ≈ {int(round(cm_distance))}cm"
        else:
            label = f"{pixel_distance:.0f}px"

        cv2.putText(frame_rgb, label, ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Convert to ImageTk
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

def on_click(event):
    global PIXEL_DISTANCE, PIXELS_PER_CM

    x = event.x
    y = event.y
    clicked_points.append((x, y))

    if len(clicked_points) == 2:
        pt1, pt2 = clicked_points
        PIXEL_DISTANCE = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        print(f"🧮 Pixel Distance: {PIXEL_DISTANCE:.2f} px")

        if PIXELS_PER_CM is None:
            real_cm = simpledialog.askfloat("Calibration",
                        f"📐 How many real-world cm is {PIXEL_DISTANCE:.2f} pixels?")
            if real_cm:
                PIXELS_PER_CM = PIXEL_DISTANCE / real_cm
                print(f"✅ Calibrated: {PIXELS_PER_CM:.2f} px/cm")
                save_calibration(PIXELS_PER_CM)
            else:
                print("❌ Calibration cancelled.")
        else:
            cm_distance = PIXEL_DISTANCE / PIXELS_PER_CM
            print(f"📏 {PIXEL_DISTANCE:.1f}px = {int(round(cm_distance))}cm")

        clicked_points.clear()

# === Tkinter GUI ===
root = tk.Tk()
root.title("🧮 Webcam Ruler (Click 2 Points to Measure)")

video_label = tk.Label(root)
video_label.pack()

instruction = tk.Label(root, text="🖱 Click two points to measure distance (auto-loads saved calibration).",
                       font=("Arial", 12))
instruction.pack(pady=5)

video_label.bind("<Button-1>", on_click)

# Start updating webcam feed
load_calibration()
update_frame()

# Exit cleanly on close
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
