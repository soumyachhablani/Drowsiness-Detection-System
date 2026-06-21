"""
Driver Drowsiness Detection System
----------------------------------
Real-time drowsiness detection using Eye Aspect Ratio (EAR) computed from
MediaPipe FaceLandmarker landmarks, with a hardware buzzer alarm on Raspberry Pi.

- Detects prolonged eye closure (drowsiness) AND prolonged loss of the face
  (head drop / looking away), and sounds an alarm for either.
- Auto-calibrates to the user's own open-eye baseline at startup.
- Runs on a laptop (visual only) or a Raspberry Pi (visual + GPIO buzzer).

Run:  python drowsiness_detector.py     (press 'q' to quit)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time

# --- Optional hardware buzzer (Raspberry Pi). Falls back gracefully on a laptop. ---
try:
    from gpiozero import PWMOutputDevice
    buzzer = PWMOutputDevice(18, frequency=2700)   # passive buzzer on GPIO 18
    buzzer.value = 0
    HAS_BUZZER = True
except Exception:
    buzzer = None
    HAS_BUZZER = False


def alarm(on):
    """Turn the buzzer on/off. value=0.5 drives a tone, value=0 forces silence."""
    if HAS_BUZZER:
        buzzer.value = 0.5 if on else 0


# --- Load the MediaPipe face-landmark model ---
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 6 landmark points per eye, ordered for the EAR formula (p1..p6)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def ear(landmarks, idx, w, h):
    """Eye Aspect Ratio = (sum of vertical eyelid gaps) / (2 * horizontal eye width)."""
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
    p1, p2, p3, p4, p5, p6 = p
    return (math.dist(p2, p6) + math.dist(p3, p5)) / (2 * math.dist(p1, p4))


# --- Tunable settings ---
CALIB_SECONDS = 5          # seconds spent learning your open-eye baseline
DROWSY_SECONDS = 2.0       # eyes shut this long -> drowsy
NO_FACE_SECONDS = 2.0      # face gone this long -> alarm
THRESHOLD_FACTOR = 0.75    # drowsy threshold = baseline EAR * this

# --- State ---
calibrating = True
calib_values = []
calib_start = None
threshold = None
closed_since = None
face_lost_since = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe expects RGB; OpenCV gives BGR.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        face_lost_since = None
        h, w = frame.shape[:2]
        lm = result.face_landmarks[0]
        avg = (ear(lm, RIGHT_EYE, w, h) + ear(lm, LEFT_EYE, w, h)) / 2

        # draw the eye points
        for i in RIGHT_EYE + LEFT_EYE:
            cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 2, (0, 255, 0), -1)

        if calibrating:
            # start the calibration clock only once the face actually appears
            if calib_start is None:
                calib_start = time.time()
            calib_values.append(avg)
            cv2.putText(frame, "CALIBRATING - keep eyes open", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if time.time() - calib_start >= CALIB_SECONDS and calib_values:
                threshold = (sum(calib_values) / len(calib_values)) * THRESHOLD_FACTOR
                calibrating = False
        else:
            cv2.putText(frame, f"EAR: {avg:.2f}  thr: {threshold:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if avg < threshold:                 # eyes are shut right now
                if closed_since is None:
                    closed_since = time.time()  # remember when they shut
                closed_for = time.time() - closed_since
                cv2.putText(frame, f"Eyes closed: {closed_for:.1f}s", (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                if closed_for >= DROWSY_SECONDS:
                    cv2.putText(frame, "DROWSY!", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    alarm(True)
                else:
                    alarm(False)
            else:                               # eyes open -> reset
                closed_since = None
                alarm(False)

    else:
        # No face detected
        closed_since = None
        if calibrating:
            cv2.putText(frame, "CALIBRATING - show your face", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            alarm(False)
        else:
            if face_lost_since is None:
                face_lost_since = time.time()
            face_lost_for = time.time() - face_lost_since
            cv2.putText(frame, f"NO FACE: {face_lost_for:.1f}s", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            if face_lost_for >= NO_FACE_SECONDS:
                cv2.putText(frame, "FACE LOST!", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                alarm(True)
            else:
                alarm(False)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

alarm(False)
cap.release()
cv2.destroyAllWindows()
