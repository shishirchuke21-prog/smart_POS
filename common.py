"""Shared helpers: face detector + AdaFace embedder, and the saved-faces database."""
import os
import pickle

import numpy as np
from insightface.app import FaceAnalysis

from adaface import AdaFace

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_db.pkl")
MODEL_TAG = "adaface_ir101_webface12m"
MATCH_THRESHOLD = 0.35  # cosine similarity; higher = stricter


class FaceSystem:
    """Detect faces with InsightFace SCRFD, embed them with AdaFace IR-101."""

    def __init__(self):
        self.detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                     providers=["CPUExecutionProvider"])
        self.detector.prepare(ctx_id=-1, det_size=(640, 640))
        self.embedder = AdaFace()

    def get(self, img):
        faces = self.detector.get(img)
        for f in faces:
            f.embedding = self.embedder.embed(img, f.kps)
        return faces


def load_model():
    return FaceSystem()


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            data = pickle.load(f)
        if data.get("model") == MODEL_TAG:
            return data["people"]
        print("Saved faces were made with a different model - please re-enroll.")
    return {}  # name -> normalized embedding


def save_db(people):
    with open(DB_PATH, "wb") as f:
        pickle.dump({"model": MODEL_TAG, "people": people}, f)


def normalize(v):
    return v / np.linalg.norm(v)


def best_match(embedding, db):
    """Return (name, score) of the closest saved face, or ("Unknown", score)."""
    if not db:
        return "Unknown", 0.0
    emb = normalize(embedding)
    names = list(db.keys())
    scores = np.array([float(np.dot(emb, db[n])) for n in names])
    i = int(scores.argmax())
    if scores[i] >= MATCH_THRESHOLD:
        return names[i], float(scores[i])
    return "Unknown", float(scores[i])
