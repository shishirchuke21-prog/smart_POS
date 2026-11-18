"""Remember a person from the webcam. Usage: python enroll.py Name

Press SPACE to capture a sample (take ~5 from slightly different angles),
then press 'q' to save.
"""
import sys

import cv2
import numpy as np

from common import load_db, load_model, normalize, save_db

if len(sys.argv) < 2:
    raise SystemExit("Usage: python enroll.py <name>")
name = sys.argv[1]

app = load_model()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (check camera permission)")

samples = []
print("SPACE = capture sample, q = save and quit")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    faces = app.get(frame)
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{name}: {len(samples)} samples", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Enroll", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        if len(faces) == 1:
            samples.append(normalize(faces[0].embedding))
            print(f"Captured sample {len(samples)}")
        else:
            print(f"Need exactly 1 face in view (saw {len(faces)})")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if samples:
    db = load_db()
    db[name] = normalize(np.mean(samples, axis=0))
    save_db(db)
    print(f"Saved {name} ({len(samples)} samples). Known people: {list(db)}")
else:
    print("No samples captured, nothing saved.")
