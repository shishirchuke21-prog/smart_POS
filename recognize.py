"""Live face recognition from the webcam. Press 'q' to quit."""
import cv2

from common import best_match, load_db, load_model

db = load_db()
print("Known people:", list(db) or "none - run enroll.py first")

app = load_model()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (check camera permission)")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    for f in app.get(frame):
        name, score = best_match(f.embedding, db)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Recognize", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
