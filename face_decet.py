"""
Multi-person GPU face recognition (PyTorch): MTCNN + FaceNet on CUDA.

Install (keeps your existing CUDA torch untouched):
  pip install facenet-pytorch --no-deps
  pip install requests tqdm opencv-python

Usage:
  python face_decet.py enroll   # S = save sample, Q = done, then type the person's name
  python face_decet.py run      # live recognition
  python face_decet.py check    # how similar are the enrolled people to each other?
  python face_decet.py delete   # remove one person
  python face_decet.py list     # show enrolled people

NOTE: this version uses margin crops + fp32, so embeddings differ from older versions.
Delete the old faces_gpu.pkl and enroll again.
"""
import os
import sys
import time
import pickle
import threading
from collections import deque, Counter

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

DB_FILE = "faces_gpu.pkl"      # {"people": {name: array(N, 512)}}
THRESHOLD = 0.65               # min cosine similarity to accept a person
MARGIN = 0.08                  # winner must beat runner-up by this much
TOP_K = 3                      # person score = mean of their K best-matching samples
VOTE_LEN = 7                   # frames remembered per face
VOTE_NEEDED = 5                # frames that must agree before a name is confirmed
CROP_MARGIN = 0.20             # extra context around the detected face box
SIMILAR_WARN = 0.70            # warn at enrollment if two people are this similar
COOLDOWN = 3                   # seconds between repeated prints per person
CAM_INDEX = 0
DETECT_WIDTH = 480             # frame is shrunk to this width for detection (speed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
if device.type != "cuda":
    print("WARNING: CUDA not available -> running on CPU (slow). Install a CUDA build of PyTorch.")
torch.backends.cudnn.benchmark = True

mtcnn = MTCNN(keep_all=True, device=device, min_face_size=60, factor=0.8, post_process=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)   # fp32 for accuracy


# ---------------------------------------------------------------- database
def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "rb") as f:
        data = pickle.load(f)
    if "people" in data:                                  # multi-person format
        return data["people"]
    if "name" in data and "embeddings" in data:           # old single-person format
        return {data["name"]: data["embeddings"]}
    return {}


def save_db(people):
    with open(DB_FILE, "wb") as f:
        pickle.dump({"people": people}, f)


def on_face_detected(name):
    # FUTURE: replace with attendance/database logic (mark `name` present once per day)
    print(f"{name} face detected")


def centroid(emb):
    c = emb.mean(axis=0)
    return c / np.linalg.norm(c)


# ---------------------------------------------------------------- camera
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


# ---------------------------------------------------------------- model
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
        if p is None or p < 0.92:
            continue
        x1, y1, x2, y2 = (box / scale).astype(int)
        if x2 - x1 < 40 or y2 - y1 < 40:
            continue
        # expand the crop a bit so the embedding sees the whole face
        mw, mh = int((x2 - x1) * CROP_MARGIN), int((y2 - y1) * CROP_MARGIN)
        cx1, cy1 = max(x1 - mw, 0), max(y1 - mh, 0)
        cx2, cy2 = min(x2 + mw, w), min(y2 + mh, h)
        crop = cv2.resize(rgb_full[cy1:cy2, cx1:cx2], (160, 160))
        crops.append(crop)
        good_boxes.append((max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)))
    if not crops:
        return [], None

    batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().to(device)
    batch = (batch - 127.5) / 128.0
    emb = torch.nn.functional.normalize(resnet(batch), dim=1)
    return good_boxes, emb


def draw_fps(frame, fps):
    text = f"FPS: {fps:.0f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = frame.shape[1] - tw - 15
    color = (0, 255, 0) if fps >= 30 else (0, 165, 255)
    cv2.putText(frame, text, (x, th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


# ---------------------------------------------------------------- matching
def person_scores(emb, people_t):
    """emb: (F,512) tensor. Returns (F,P) score per face per person (mean of top-K sims)."""
    cols = []
    for E in people_t:
        sims = emb @ E.T                                   # (F, N)
        k = min(TOP_K, sims.shape[1])
        cols.append(sims.topk(k, dim=1).values.mean(dim=1))
    return torch.stack(cols, dim=1).cpu().numpy()


def decide(scores_row, names):
    """Returns (candidate_name_or_None, best_score)."""
    order = np.argsort(scores_row)[::-1]
    best = scores_row[order[0]]
    if best < THRESHOLD:
        return None, best
    if len(order) > 1 and best - scores_row[order[1]] < MARGIN:
        return None, best                                  # too close to call
    return names[order[0]], best


class Tracker:
    """Follows each face across frames and confirms a name only after repeated agreement."""

    def __init__(self):
        self.tracks = []

    def update(self, boxes, candidates):
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t["t"] < 1.0]
        used, confirmed = set(), []
        for box, cand in zip(boxes, candidates):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best, bd = None, 1e9
            for t in self.tracks:
                if id(t) in used:
                    continue
                d = ((t["c"][0] - cx) ** 2 + (t["c"][1] - cy) ** 2) ** 0.5
                if d < bd:
                    best, bd = t, d
            if best is None or bd > 100:
                best = {"hist": deque(maxlen=VOTE_LEN)}
                self.tracks.append(best)
            best["c"], best["t"] = (cx, cy), now
            best["hist"].append(cand or "Unknown")
            used.add(id(best))
            top, n = Counter(best["hist"]).most_common(1)[0]
            confirmed.append(top if top != "Unknown" and n >= VOTE_NEEDED else None)
        return confirmed


# ---------------------------------------------------------------- enroll
def capture_samples(cam):
    samples = []
    print("Look at the camera. Press S to save a sample (~30-50, vary angle/lighting). Q to finish.")
    print("Make sure ONLY ONE person is in front of the camera.")
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
        cv2.imshow("Enroll (S = save, Q = done)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if emb is not None and len(boxes) == 1:
                samples.append(emb[0].cpu().numpy())
                print(f"Saved sample {len(samples)}")
            else:
                print("Need exactly one face in frame.")
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return samples


def enroll():
    cam = Camera(CAM_INDEX)
    try:
        while True:
            samples = capture_samples(cam)
            if not samples:
                print("No samples captured, nothing saved.")
            else:
                name = input(f"Enter a name for these {len(samples)} samples: ").strip()
                if not name:
                    print("Empty name, nothing saved.")
                else:
                    people = load_db()
                    key = next((k for k in people if k.lower() == name.lower()), name)
                    new = np.stack(samples)

                    # safety check: does this face look like someone already enrolled?
                    c_new = centroid(new)
                    risky = [(k, float(c_new @ centroid(v))) for k, v in people.items() if k != key]
                    risky = [(k, s) for k, s in risky if s >= SIMILAR_WARN]
                    save = True
                    for k, s in risky:
                        print(f"WARNING: these samples look very similar to '{k}' (similarity {s:.2f}).")
                        print("  Different people normally score below 0.5. Is this the same person, "
                              "or was someone else in frame?")
                    if risky:
                        save = input("Save anyway? (y/n): ").strip().lower() == "y"

                    if save:
                        if key in people:
                            people[key] = np.concatenate([people[key], new])
                            print(f"Added to existing '{key}' (now {len(people[key])} samples).")
                        else:
                            people[key] = new
                            print(f"Saved new person '{key}' ({len(new)} samples).")
                        save_db(people)
                    else:
                        print("Not saved.")

            if input("Enroll another person? (y/n): ").strip().lower() != "y":
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
    print("Enrolled people:", ", ".join(load_db().keys()) or "none")


# ---------------------------------------------------------------- run
def run():
    people = load_db()
    if not people:
        print("No one is enrolled yet. Run enroll first.")
        return
    names = list(people.keys())
    print("Loaded people:", ", ".join(f"{n} ({len(e)})" for n, e in people.items()))
    people_t = [torch.from_numpy(people[n]).float().to(device) for n in names]

    cam = Camera(CAM_INDEX)
    tracker = Tracker()
    last_print = {}
    fps, last = 0.0, time.time()
    while True:
        frame = cam.read()
        if frame is None:
            continue

        boxes, emb = detect_and_embed(frame)
        if emb is not None:
            scores = person_scores(emb, people_t)
            decisions = [decide(row, names) for row in scores]
            candidates = [d[0] for d in decisions]
            confirmed = tracker.update(boxes, candidates)

            for (x1, y1, x2, y2), (cand, best), conf in zip(boxes, decisions, confirmed):
                if conf:
                    label, color = f"{conf} ({best:.2f})", (0, 255, 0)
                elif cand:
                    label, color = f"checking... ({best:.2f})", (0, 255, 255)
                else:
                    label, color = f"Unknown ({best:.2f})", (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if conf and time.time() - last_print.get(conf, 0) > COOLDOWN:
                    on_face_detected(conf)
                    last_print[conf] = time.time()

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
        last = now
        draw_fps(frame, fps)
        cv2.imshow("Recognition (Q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- check / delete / list
def check():
    people = load_db()
    if len(people) < 2:
        print("Need at least 2 enrolled people to compare.")
        return
    names = list(people.keys())
    cents = {n: centroid(people[n]) for n in names}

    print("\nWithin each person (how consistent are their own samples?):")
    for n in names:
        sims = people[n] @ cents[n]
        print(f"  {n:15s} samples={len(sims):3d}  avg={sims.mean():.2f}  min={sims.min():.2f}")

    print("\nBetween people (should be LOW, ideally < 0.5):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            c_sim = float(cents[a] @ cents[b])
            cross = float((people[a] @ people[b].T).max())
            flag = "  <-- TOO SIMILAR: bad data? re-enroll one of them" if c_sim >= SIMILAR_WARN else ""
            print(f"  {a} vs {b}: centroid={c_sim:.2f}  closest-pair={cross:.2f}{flag}")


def delete():
    people = load_db()
    if not people:
        print("No one enrolled.")
        return
    print("Enrolled:", ", ".join(people.keys()))
    name = input("Name to delete: ").strip()
    key = next((k for k in people if k.lower() == name.lower()), None)
    if key is None:
        print("Not found.")
    elif input(f"Delete '{key}' and all {len(people[key])} samples? (y/n): ").strip().lower() == "y":
        del people[key]
        save_db(people)
        print("Deleted.")


def list_people():
    people = load_db()
    if people:
        for n, e in people.items():
            print(f"{n}: {len(e)} samples")
    else:
        print("No one enrolled yet.")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    modes = {"enroll": enroll, "run": run, "check": check, "delete": delete, "list": list_people}
    if len(sys.argv) == 2 and sys.argv[1] in modes:
        mode = sys.argv[1]
    else:
        mode = input("Type 'enroll', 'run', 'check', 'delete' or 'list': ").strip().lower()
    modes.get(mode, lambda: print("Invalid choice."))()