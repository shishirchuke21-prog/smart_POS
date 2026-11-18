"""Detect faces in a photo. Usage: python detect.py photo.jpg"""
import sys

import cv2

from common import load_model

path = sys.argv[1] if len(sys.argv) > 1 else "me.jpg"
img = cv2.imread(path)
if img is None:
    raise SystemExit(f"Could not read image: {path}")

app = load_model()
faces = app.get(img)
print("Faces found:", len(faces))

for f in faces:
    x1, y1, x2, y2 = f.bbox.astype(int)
    print("Box:", (int(x1), int(y1), int(x2), int(y2)), "Embedding size:", f.embedding.shape)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("detected.jpg", img)
print("Saved boxes to detected.jpg")
