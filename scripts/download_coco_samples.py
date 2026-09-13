import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMG_DIR = DATA / "images"
ANN_DIR = DATA / "coco"
IMG_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)

ANIMAL_NAMES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
}

def download(url, path):
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)

def main(count):
    ann_zip = DATA / "annotations_trainval2017.zip"
    #download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)
    #with zipfile.ZipFile(ann_zip) as z:
    #    z.extract("annotations/instances_val2017.json", DATA)

    ann_path = DATA / "annotations/instances_val2017.json"
    with ann_path.open() as f:
        coco = json.load(f)

    names = {c["id"]: c["name"] for c in coco["categories"]}
    image_names = {x["id"]: x["file_name"] for x in coco["images"]}
    by_image = defaultdict(list)

    for a in coco["annotations"]:
        category_name = names[a["category_id"]]
        if category_name in ANIMAL_NAMES and a.get("iscrowd", 0) == 0:
            by_image[a["image_id"]].append(category_name)

    #selected = [(i, labels) for i, labels in by_image.items() if len(labels) >= 2]
    selected = [(i, labels) for i, labels in by_image.items() if len(labels) >= 2][:count]

    for image_id, labels in selected:
        name = image_names[image_id]
        path = IMG_DIR / name
        download(f"http://images.cocodataset.org/val2017/{name}", path)
        print(name, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    main(args.count)
