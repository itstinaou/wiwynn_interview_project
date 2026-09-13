import os
import math
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Sam2Processor, Sam2Model

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/images"
OUTPUT_DIR = ROOT / "outputs"
VIS_DIR = OUTPUT_DIR / "visualizations"
MASK_DIR = OUTPUT_DIR / "masks"
for p in (OUTPUT_DIR, VIS_DIR, MASK_DIR):
    p.mkdir(parents=True, exist_ok=True)

ANIMAL_LABELS = ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

def get_device():
    requested = os.getenv("MODEL_DEVICE", "")
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
ANIMAL_THRESHOLD = float(os.getenv("ANIMAL_THRESHOLD", "0.1"))
EYE_THRESHOLD = float(os.getenv("EYE_THRESHOLD", "0.12"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.20"))

_dino_processor = _dino_model = _sam_processor = _sam_model = None

def load_models():
    global _dino_processor, _dino_model, _sam_processor, _sam_model
    if _dino_model is None:
        _dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(DEVICE).eval()
    if _sam_model is None:
        _sam_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
        _sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(DEVICE).eval()

def detect(image, labels, threshold):
    text = ". ".join(f"a {x}" for x in labels) + "."
    inputs = _dino_processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = _dino_model(**inputs)
    result = _dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)]
    )[0]
    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    labels_out = result.get("text_labels", ["object"] * len(boxes))
    detections = []
    for box, score, label in zip(boxes, scores, labels_out):
        detections.append({"box": box.tolist(), "score": float(score), "label": str(label).lower()})
    return detections

def segment_box(image, box):
    x1, y1, x2, y2 = map(float, box)
    inputs = _sam_processor(images=image, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
    inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = _sam_model(**inputs, multimask_output=False)
    masks = _sam_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0]
    return masks[0, 0].numpy().astype(np.uint8)

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def nms(boxes, iou_threshold=0.5):
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    kept = []

    while boxes:
        current = boxes.pop(0)
        kept.append(current)
        boxes = [x for x in boxes if box_iou(current["box"], x["box"]) < iou_threshold]

    return kept


def detect_eyes(image, animal):
    ax1, ay1, ax2, ay2 = map(int, animal["box"])
    ax1, ay1 = max(0, ax1), max(0, ay1)
    ax2, ay2 = min(image.width, ax2), min(image.height, ay2)

    crop = image.crop((ax1, ay1, ax2, ay2))
    candidates = detect(crop, ["eye"], EYE_THRESHOLD)

    animal_w = ax2 - ax1
    animal_h = ay2 - ay1
    animal_box = [0, 0, animal_w, animal_h]

    valid_candidates = []
    for candidate in candidates:
        box = candidate["box"]
        if box_iou(box, animal_box) < 0.5:
            valid_candidates.append(candidate)

    valid_candidates = nms(valid_candidates)[:2]

    eyes = []
    for candidate in valid_candidates:
        bx1, by1, bx2, by2 = candidate["box"]
        global_box = [bx1 + ax1, by1 + ay1, bx2 + ax1, by2 + ay1]

        mask = segment_box(image, global_box)
        center = centroid(mask)

        if center is not None:
            eyes.append({
                "box": global_box,
                "mask": mask,
                "center": center,
                "score": candidate["score"]
            })

    eyes.sort(key=lambda x: x["center"][0])
    return eyes

def draw_result(image, animals):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for animal in animals:
        mask = animal["mask"].astype(bool)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (255, 180, 0), 2)
        x1, y1, x2, y2 = map(int, animal["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.putText(frame, f'{animal["id"]}: {animal["label"]}', (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

        for j, eye in enumerate(animal["eyes"]):
            mask = eye["mask"].astype(bool)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            cx, cy = map(int, eye["center"])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, "L" if j == 0 else "R", (cx + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if len(animal["eyes"]) == 2:
            p1, p2 = animal["eyes"][0]["center"], animal["eyes"][1]["center"]
            cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 255), 2)
            mid = ((int(p1[0] + p2[0]) / 2), (int(p1[1] + p2[1]) / 2))
            cv2.putText(frame, f'{distance(p1, p2):.1f}px', (int(mid[0]), int(mid[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame

def run_image(path, animal_classes):
    load_models()
    image = Image.open(path).convert("RGB")
    animals = detect(image, animal_classes, ANIMAL_THRESHOLD)
    print("DINO detections:", animals)
    animals = [x for x in animals if any(c in x["label"].lower() for c in animal_classes)][:6]
    print("Animals:", animals)

    for i, animal in enumerate(animals, 1):
        animal["id"] = i
        animal["mask"] = segment_box(image, animal["box"])
        cv2.imwrite(str(MASK_DIR / f"{Path(path).stem}_animal_{i}.png"), animal["mask"] * 255)
        animal["eyes"] = detect_eyes(image, animal)
        for j, eye in enumerate(animal["eyes"], 1):
            cv2.imwrite(str(MASK_DIR / f"{Path(path).stem}_animal_{i}_eye_{j}.png"), eye["mask"] * 255)

    valid_right_eyes = [(a["id"], a["eyes"][1]["center"]) for a in animals if len(a["eyes"]) == 2]
    pair_distance = None
    pair_ids = None
    if len(valid_right_eyes) >= 2:
        pair_ids = (valid_right_eyes[0][0], valid_right_eyes[1][0])
        pair_distance = distance(valid_right_eyes[0][1], valid_right_eyes[1][1])

    rows = []
    for animal in animals:
        eyes = animal["eyes"]
        row = {
            "image_id": Path(path).stem,
            "animal_id": animal["id"],
            "animal_label": animal["label"],
            "left_eye_x": eyes[0]["center"][0] if len(eyes) > 0 else None,
            "left_eye_y": eyes[0]["center"][1] if len(eyes) > 0 else None,
            "right_eye_x": eyes[1]["center"][0] if len(eyes) > 1 else None,
            "right_eye_y": eyes[1]["center"][1] if len(eyes) > 1 else None,
            "eye_distance_px": distance(eyes[0]["center"], eyes[1]["center"]) if len(eyes) == 2 else None,
            "right_eye_pair_animal_1": pair_ids[0] if pair_ids else None,
            "right_eye_pair_animal_2": pair_ids[1] if pair_ids else None,
            "right_eye_pair_distance_px": pair_distance
        }
        rows.append(row)

    cv2.imwrite(str(VIS_DIR / f"{Path(path).stem}.jpg"), draw_result(image, animals))
    return rows

def run_all(animal_classes):
    paths = sorted(IMAGE_DIR.glob("*.jpg"))
    all_rows = []
    for path in paths:
        print(f"Processing {path.name}")
        try:
            all_rows.extend(run_image(path, animal_classes))
        except Exception as e:
            print(f"Failed: {path.name}: {e}")
    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / "results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animals", nargs="+", default=["cat"], choices=ANIMAL_LABELS)
    args = parser.parse_args()
    run_all(args.animals)
