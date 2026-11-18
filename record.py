"""Record video from the webcam to recording.mp4. Press 'q' to stop."""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam (check camera permission)")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("recording.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))

print("Recording... press 'q' to stop")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved to recording.mp4")
