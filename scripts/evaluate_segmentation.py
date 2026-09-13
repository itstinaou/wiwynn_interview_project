import csv
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs/masks"
OUT = ROOT / "outputs/segmentation_evaluation.csv"

def score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    return iou, dice

def main():
    pairs = []
    for gt_path in (ROOT / "data/ground_truth_masks").glob("*.png"):
        pred_path = PRED / gt_path.name
        if pred_path.exists():
            pred = np.array(Image.open(pred_path))
            gt = np.array(Image.open(gt_path))
            iou, dice = score(pred, gt)
            pairs.append({"mask": gt_path.name, "iou": iou, "dice": dice})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mask", "iou", "dice"])
        w.writeheader()
        w.writerows(pairs)
    print(f"Evaluated {len(pairs)} masks")

if __name__ == "__main__":
    main()
