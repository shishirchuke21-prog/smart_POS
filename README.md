# Face Project

Detect faces with InsightFace SCRFD, recognize them with **AdaFace IR-101** (WebFace12M) from
Hugging Face: `minchul/cvlface_adaface_ir101_webface12m` (code in `adaface.py`, weights in `models_cache/`).

| Command | What it does |
|---|---|
| `./run.sh record` | Record webcam to `recording.mp4` (q to stop) |
| `./run.sh detect photo.jpg` | Find faces in a photo, saves `detected.jpg` |
| `./run.sh enroll Alice` | Remember a person: SPACE to capture ~5 samples, q to save |
| `./run.sh recognize` | Live webcam recognition with names (q to quit) |
| `./run.sh forget [name]` | List saved people, or remove one |

Saved faces live in `faces_db.pkl`. Tune `MATCH_THRESHOLD` in `common.py`
(default 0.35; raise it if strangers get matched, lower it if you aren't recognized).

Setup (already done): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

Note: both buffalo_l and the WebFace12M-trained AdaFace weights are for non-commercial/research use.
