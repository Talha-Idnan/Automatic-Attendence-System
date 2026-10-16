"""
GPU face recognition (PyTorch): MTCNN detection + FaceNet embeddings on CUDA.

Install (keeps your existing CUDA torch untouched):
  pip install facenet-pytorch --no-deps
  pip install requests tqdm opencv-python

Usage:
  python face_gpu.py enroll   # press S to save a sample (~20), Q to finish
  python face_gpu.py run      # live recognition, Q to quit
"""
import sys
import time
import pickle
import threading

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

NAME = "Talha"
DB_FILE = "faces_gpu.pkl"      # new file: embeddings differ from the old dlib ones
THRESHOLD = 0.65               # cosine similarity; higher = stricter
COOLDOWN = 3                   # seconds between repeated prints
CAM_INDEX = 0
DETECT_WIDTH = 480             # frame is shrunk to this width for detection (speed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
if device.type != "cuda":
    print("WARNING: CUDA not available -> running on CPU (slow). Install a CUDA build of PyTorch.")
torch.backends.cudnn.benchmark = True

use_half = device.type == "cuda"
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=60, factor=0.8, post_process=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
if use_half:
    resnet = resnet.half()


def on_face_detected(name):
    # FUTURE: replace with attendance/database logic
    print(f"{name} face detected")


class Camera:
    """Reads frames in a background thread so the main loop never waits on the webcam."""

    def __init__(self, index=0):
        self.cap = None
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            for idx in (index, 1, 2):
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        print(f"Camera opened: index {idx}, backend {backend}")
                        self.cap = cap
                        break
                cap.release()
            if self.cap:
                break
        if self.cap is None:
            raise RuntimeError(
                "No camera found. Check: webcam plugged in, Windows camera privacy "
                "settings, and that no other app (Zoom/Teams/browser) is using it."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f

    def read(self):
        return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


@torch.no_grad()
def detect_and_embed(frame):
    """Returns (boxes in original-frame coords, L2-normalized embeddings) for all faces."""
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / w
    small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb_small)
    if boxes is None:
        return [], None

    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crops, good_boxes = [], []
    for box, p in zip(boxes, probs):
        if p is None or p < 0.9:
            continue
        x1, y1, x2, y2 = (box / scale).astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = cv2.resize(rgb_full[y1:y2, x1:x2], (160, 160))
        crops.append(crop)
        good_boxes.append((x1, y1, x2, y2))
    if not crops:
        return [], None

    batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(device)
    batch = (batch - 127.5) / 128.0
    if use_half:
        batch = batch.half()
    emb = resnet(batch).float()
    emb = torch.nn.functional.normalize(emb, dim=1)
    return good_boxes, emb


def draw_fps(frame, fps):
    text = f"FPS: {fps:.0f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = frame.shape[1] - tw - 15
    color = (0, 255, 0) if fps >= 30 else (0, 165, 255)
    cv2.putText(frame, text, (x, th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def enroll():
    cam = Camera(CAM_INDEX)
    samples = []
    print("Look at the camera. Press S to save a sample (~20, vary angle/lighting). Q to finish.")
    fps, last = 0.0, time.time()
    while True:
        frame = cam.read()
        if frame is None:
            continue
        boxes, emb = detect_and_embed(frame)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {len(samples)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
        last = now
        draw_fps(frame, fps)
        cv2.imshow("Enroll", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if emb is not None and len(boxes) == 1:
                samples.append(emb[0].cpu().numpy())
                print(f"Saved sample {len(samples)}")
            else:
                print("Need exactly one face in frame.")
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    if samples:
        with open(DB_FILE, "wb") as f:
            pickle.dump({"name": NAME, "embeddings": np.stack(samples)}, f)
        print(f"Saved {len(samples)} samples to {DB_FILE}")


def run():
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
    name = db["name"]
    known = torch.from_numpy(db["embeddings"]).float().to(device)  # (K, 512)

    cam = Camera(CAM_INDEX)
    last_print = 0
    fps, last = 0.0, time.time()
    while True:
        frame = cam.read()
        if frame is None:
            continue

        boxes, emb = detect_and_embed(frame)
        if emb is not None:
            sims = (emb @ known.T).max(dim=1).values.cpu().numpy()  # best cosine sim per face
            for (x1, y1, x2, y2), s in zip(boxes, sims):
                matched = s >= THRESHOLD
                color = (0, 255, 0) if matched else (0, 0, 255)
                label = f"{name} ({s:.2f})" if matched else f"Unknown ({s:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if matched and time.time() - last_print > COOLDOWN:
                    on_face_detected(name)
                    last_print = time.time()

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
        last = now
        draw_fps(frame, fps)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("enroll", "run"):
        mode = sys.argv[1]
    else:
        mode = input("Type 'enroll' or 'run': ").strip().lower()

    if mode == "enroll":
        enroll()
    elif mode == "run":
        run()
    else:
        print("Invalid choice.")