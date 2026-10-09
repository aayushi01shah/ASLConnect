"""Record training samples for ASL letters.

Run:  python collect_data.py
Press a letter key (A-Y, except J) while making that sign. The script grabs
SAMPLES_PER_BURST frames of your hand shape. Slowly tilt and shift your hand
while it records so the model sees some variety. Press ESC to quit.
Samples are appended to data/landmarks.csv.
"""
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from drawLandmarks import draw_landmarks_on_image
from imgHandling import collectFrame, convertToMpImage
from letterFeatures import extractFeatures, LETTERS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "data", "landmarks.csv")
SAMPLES_PER_BURST = 60


def loadCounts():
    counts = Counter()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, newline="") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] += 1
    return counts


def main():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(
            model_asset_path=os.path.join(HERE, "gesture_recognizer.task")))
    recognizer = vision.GestureRecognizer.create_from_options(options)

    counts = loadCounts()
    cap = cv2.VideoCapture(0)
    activeLetter, remaining = None, 0
    message = "Press a letter key (A-Y, no J) to record it. ESC quits."

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        while cap.isOpened():
            frameForDetection, frameForOutput = collectFrame(capIn=cap)
            results = recognizer.recognize(convertToMpImage(frameForDetection))
            frame = draw_landmarks_on_image(frameForOutput, results)

            if remaining > 0 and results.hand_landmarks:
                feats = extractFeatures(results.hand_landmarks[0],
                                        results.handedness[0][0].category_name)
                if feats is not None:
                    writer.writerow([activeLetter] + [f"{v:.5f}" for v in feats])
                    counts[activeLetter] += 1
                    remaining -= 1
                    if remaining == 0:
                        f.flush()
                        message = f"Saved {activeLetter}. Total {counts[activeLetter]}. Press another letter."
            elif remaining > 0:
                message = "No hand detected - show your hand"

            if remaining > 0 and results.hand_landmarks:
                message = f"Recording {activeLetter}: {SAMPLES_PER_BURST - remaining}/{SAMPLES_PER_BURST}"

            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            summary = " ".join(f"{k}:{counts[k]}" for k in sorted(counts))
            cv2.putText(frame, summary[-90:], (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Collect ASL data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if remaining == 0 and 97 <= key <= 122:
                letter = chr(key).upper()
                if letter in LETTERS:
                    activeLetter, remaining = letter, SAMPLES_PER_BURST
                else:
                    message = f"{letter} needs motion (J, Z) - not supported"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
