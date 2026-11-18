"""AdaFace IR-101 (WebFace12M) face embedder from Hugging Face:
https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m
"""
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from insightface.utils import face_align
from omegaconf import OmegaConf

REPO_ID = "minchul/cvlface_adaface_ir101_webface12m"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache", "adaface_ir101")


class AdaFace:
    def __init__(self):
        if not os.path.exists(os.path.join(LOCAL_DIR, "pretrained_model", "model.pt")):
            print("Downloading AdaFace IR-101 (~260 MB)...")
            snapshot_download(REPO_ID, local_dir=LOCAL_DIR,
                              allow_patterns=["models/*", "models/**", "pretrained_model/*"])
        sys.path.insert(0, LOCAL_DIR)
        from models import get_model  # model code from the HF repo
        sys.path.pop(0)

        with open(os.path.join(LOCAL_DIR, "pretrained_model", "model.yaml")) as f:
            conf = OmegaConf.create(yaml.safe_load(f))
        self.model = get_model(conf)
        state = torch.load(os.path.join(LOCAL_DIR, "pretrained_model", "model.pt"),
                           map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"AdaFace weights did not load: {len(missing)} missing keys")
        self.model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, bgr_image, kps):
        """Align a face using its 5 landmarks and return a 512-d embedding."""
        aligned = face_align.norm_crop(bgr_image, landmark=kps, image_size=112)  # 112x112 BGR
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy((rgb - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(x)
        return out[0].cpu().numpy()
