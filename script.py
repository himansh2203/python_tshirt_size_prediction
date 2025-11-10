import cv2
import mediapipe as mp
import numpy as np
import math
import sys
import os
from datetime import datetime

mp_pose = mp.solutions.pose

# --- Helper functions ---
def get_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None, img
        lm = res.pose_landmarks.landmark
        h, w = img.shape[:2]
        # convert to pixel coords
        pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))}
        return pts, img

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# landmark indices used by MediaPipe Pose
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_EAR = 7
RIGHT_EAR = 8

# --- Core estimator ---
def estimate_measurements(front_img_path, side_img_path=None, height_cm=None):
    front_pts, front_img = get_landmarks(front_img_path)
    if front_pts is None:
        raise RuntimeError("No pose detected in front image. Try clearer full-body front photo.")
    # shoulder pixel distance (front)
    if LEFT_SHOULDER in front_pts and RIGHT_SHOULDER in front_pts:
        shoulder_px = distance(front_pts[LEFT_SHOULDER], front_pts[RIGHT_SHOULDER])
    else:
        raise RuntimeError("Shoulder landmarks not detected in front image.")
    # torso pixel height (shoulder to mid-hip) as reference
    mid_hip = ((front_pts[LEFT_HIP][0] + front_pts[RIGHT_HIP][0]) / 2,
               (front_pts[LEFT_HIP][1] + front_pts[RIGHT_HIP][1]) / 2)
    # mid-shoulder (use midpoint)
    mid_shoulder = ((front_pts[LEFT_SHOULDER][0] + front_pts[RIGHT_SHOULDER][0]) / 2,
                    (front_pts[LEFT_SHOULDER][1] + front_pts[RIGHT_SHOULDER][1]) / 2)
    torso_px = abs(mid_hip[1] - mid_shoulder[1])
    if torso_px <= 0:
        raise RuntimeError("Invalid torso pixel measurement.")

    # Estimate scale: convert pixels -> cm
    # We'll use anthropometric ratio: torso length (shoulder to hip) ≈ 0.30 * height (approx)
    # If height provided, use it; otherwise use default height 170 cm (warn user)
    used_height = height_cm if height_cm is not None else 170.0
    torso_real_cm = 0.30 * used_height
    scale_cm_per_px = torso_real_cm / torso_px

    shoulder_cm = shoulder_px * scale_cm_per_px

    # Convert shoulder breadth to chest circumference estimate.
    # Empirical factor: chest_circumference ≈ shoulder_breadth * 2.05 - 2.3 (varies by body type)
    # We'll use factor 2.1 as a reasonable average.
    chest_cm_est = shoulder_cm * 2.1

    # Also estimate waist from hip width (front)
    if LEFT_HIP in front_pts and RIGHT_HIP in front_pts:
        hip_px = distance(front_pts[LEFT_HIP], front_pts[RIGHT_HIP])
        hip_cm = hip_px * scale_cm_per_px
    else:
        hip_cm = None

    # If side image is provided: we can try to estimate depth (chest thickness) and refine chest circ
    depth_factor = None
    if side_img_path:
        try:
            side_pts, side_img = get_landmarks(side_img_path)
            if side_pts:
                # Use nose-to-hip vertical distance in side image as proxy scale check
                if NOSE in side_pts and LEFT_HIP in side_pts and RIGHT_HIP in side_pts:
                    side_mid_hip = ((side_pts[LEFT_HIP][0] + side_pts[RIGHT_HIP][0]) / 2,
                                    (side_pts[LEFT_HIP][1] + side_pts[RIGHT_HIP][1]) / 2)
                    side_torso_px = abs(side_mid_hip[1] - side_pts[NOSE][1])
                    # compare with front torso_px for a crude depth correction
                    if side_torso_px > 0:
                        depth_factor = torso_px / side_torso_px
                        # Don't overfit; use depth_factor to slightly adjust chest estimate:
                        chest_cm_est = chest_cm_est * (1 + 0.05 * (depth_factor - 1))
        except Exception:
            pass

    return {
        "used_height_cm": used_height,
        "shoulder_cm": round(shoulder_cm, 1),
        "chest_cm_est": round(chest_cm_est, 1),
        "hip_cm": round(hip_cm, 1) if hip_cm else None,
        "scale_cm_per_px": round(scale_cm_per_px, 4),
        "depth_factor": round(depth_factor, 3) if depth_factor else None,
    }

# --- Size mapping (men's unisex chest circumference in cm) ---
SIZE_TABLE = [
    ("XS", 76, 84),
    ("S", 84, 94),
    ("M", 94, 100),
    ("L", 100, 108),
    ("XL", 108, 116),
    ("XXL", 116, 124),
]

def chest_to_size(chest_cm):
    for label, low, high in SIZE_TABLE:
        if low <= chest_cm <= high:
            return label
    if chest_cm < SIZE_TABLE[0][1]:
        return SIZE_TABLE[0][0]
    return SIZE_TABLE[-1][0]

# --- New: capture two photos from webcam ---
def capture_two_photos(save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    captured_paths = []
    stage = 0  # 0 = front, 1 = side
    instructions = ["Front photo: press SPACE to capture, ESC to quit",
                    "Side photo: press SPACE to capture, ESC to quit"]
    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            text = instructions[stage]
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE - capture
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"capture_{'front' if stage==0 else 'side'}_{ts}.jpg"
                path = os.path.join(save_dir, fname)
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
                captured_paths.append(path)
                stage += 1
                if stage >= 2:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if len(captured_paths) < 2:
        raise RuntimeError("Two photos not captured.")
    return captured_paths[0], captured_paths[1]

# --- Command-line usage ---
def main():
    # If no args provided, open camera to capture two photos.
    if len(sys.argv) == 1:
        print("No images provided. Opening webcam to capture front and side photos.")
        front, side = capture_two_photos()

        # Ask user for height to improve estimate
        height = None
        while True:
            try:
                s = input("Enter height in cm (press Enter to use default 170): ").strip()
            except EOFError:
                # In non-interactive environments, fall back to default
                s = ""
            if s == "":
                height = None
                break
            try:
                height = float(s)
                if height <= 0:
                    print("Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter height as a number (e.g. 175).")
    else:
        if len(sys.argv) < 2:
            print("Usage: python estimate_size.py front.jpg [side.jpg] [height_cm]")
            sys.exit(1)
        front = sys.argv[1]
        side = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].isdigit() else None
        height = None
        if len(sys.argv) >= 3:
            # if 2nd arg was side image and 3rd exists and is a number -> height
            if side and len(sys.argv) >= 4 and sys.argv[3].isdigit():
                height = float(sys.argv[3])
            # if 2nd arg is actually height (number)
            elif sys.argv[2].isdigit():
                height = float(sys.argv[2])

    res = estimate_measurements(front, side, height)
    chest = res["chest_cm_est"]
    size = chest_to_size(chest)
    print("=== Measurement estimate ===")
    print(f"Used height (cm): {res['used_height_cm']}")
    print(f"Shoulder breadth (cm): {res['shoulder_cm']}")
    print(f"Estimated chest circumference (cm): {chest}")
    if res['hip_cm']:
        print(f"Hip width (cm): {res['hip_cm']}")
    if res['depth_factor']:
        print(f"Side/front depth factor: {res['depth_factor']}")
    print(f"Suggested size (approx): {size}")
    print("\nNote: This is an estimate. For best accuracy provide height and clear full-body front+side photos.")

if __name__ == "__main__":
    main()




