# Animal Segmentation & Metrology

## 1. Project Overview

This project demonstrates an end-to-end image segmentation and metrology pipeline using images selected from the COCO 2017 validation set.

For an image containing at least two animals, the system:

1. Detects the selected COCO animal classes.
2. Segments each animal individually.
3. Detects the eyes inside each animal region.
4. Segments each eye individually.
5. Uses the centroid of each eye mask as the measurement point.
6. Measures the distance between the two eyes of each animal.
7. Measures the distance between the righeyet eyes of two animals.

The demo case uses two cats with clearly visible eyes. The pipeline supports multiple animal classes through the `--animals` command-line argument.

## 2. Technology Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers
- Grounding DINO Tiny — zero-shot object detection / localization
- SAM 2.1 Hiera Tiny — segmentation from box prompts
- OpenCV
- NumPy / Pandas
- FastAPI
- Docker / Docker Compose

Grounding DINO is used to obtain object bounding boxes. SAM 2.1 converts the box prompts into segmentation masks. The same approach is used for both animals and eyes.

## 3. Project Structure

```text
.
├── src/
│   ├── pipeline.py
│   ├── api.py
│   └── __init__.py
├── scripts/
│   ├── download_coco_samples.py
│   └── evaluate_segmentation.py
├── data/
│   ├── images/
│   ├── test_data.csv
│   ├── sample_measurements.csv
│   └── eye_reference_template.csv
├── docs/
│   ├── architecture.svg
│   ├── pipeline.md
│   └── technical_notes.md
├── outputs/
│   ├── masks/
│   ├── visualizations/
│   └── results.csv
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 4. Local Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put input images under `data/images/`.

The recommended input method is `data/test_data.csv`. Each row specifies an image filename and the animal class(es) to process. Multiple rows are supported, and multiple animal classes in one row are separated by `;`.

Example:

```csv
image,animals
000000402473.jpg,cat
000000402473.jpg,"cat;dog"
```

Run the test manifest:

```bash
python -m src.pipeline
```

You can also provide a different CSV:

```bash
python -m src.pipeline --input data/test_data.csv
```

For quick direct testing, the original command-line mode is also supported:

### Run with one animal class

```bash
python -m src.pipeline --animals cat
```

### Run with multiple animal classes

```bash
python -m src.pipeline --animals cat dog
```

Supported classes:

`cat`, `dog`, `bird`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`

## 5. Outputs

After running the pipeline:

- `outputs/results.csv` — measurement results
- `outputs/visualizations/` — annotated images
- `outputs/masks/` — binary segmentation masks for animals and eyes

### CSV fields

- `image_id`: source image ID
- `animal_id`: animal index assigned by the pipeline
- `animal_label`: detected animal class
- `left_eye_x`, `left_eye_y`: centroid of the left-eye mask
- `right_eye_x`, `right_eye_y`: centroid of the right-eye mask
- `eye_distance_px`: Euclidean distance between the two eye centroids
- `right_eye_pair_animal_1`, `right_eye_pair_animal_2`: animal IDs used for the cross-animal measurement
- `right_eye_pair_distance_px`: Euclidean distance between their right-eye centroids

`data/test_data.csv` is the test input manifest. It contains the image filename and the animal class(es) to process.

`data/sample_measurements.csv` is retained as an example of measurement output.

## 6. Measurement Method

For two eye centers `(x1, y1)` and `(x2, y2)`, the image-space distance is:

```text
d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

The eye center is calculated from the predicted segmentation mask:

```text
cx = mean(x)
cy = mean(y)
```

Eyes are ordered by image-space x coordinate: the smaller x coordinate is labeled left and the larger x coordinate is labeled right. This does not assume a fixed animal pose or image orientation.

The current result is reported in pixels. Conversion to a physical unit such as millimeters requires camera calibration or a known reference scale.

## 7. Validation

For segmentation masks with ground-truth annotations:

```text
IoU = intersection / union

Dice = 2 * intersection / (predicted_area + ground_truth_area)
```

`python scripts/evaluate_segmentation.py` provides mask-level evaluation when ground-truth masks are available.

COCO does not provide eye annotations. Therefore, eye segmentation and measurement should be validated using manually annotated eye masks or eye centers on a held-out test set. Eye-center error can be reported with MAE/RMSE, while distance error can be reported with MAE, RMSE, and percentage error.

## 8. API

Start the API locally:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Interactive API documentation:

`http://localhost:8000/docs`

Health check:

```bash
curl http://localhost:8000/health
```

Measurement request:

```bash
curl -u demo:demo123 -X POST http://localhost:8000/measure
```

The `/measure` endpoint uses `data/test_data.csv` as the test input manifest and returns the measurement results as JSON.

## 9. Test Account

The local/demo API uses HTTP Basic Authentication:

```text
Username: demo
Password: demo123
```

These are demonstration credentials only. They can be changed through `.env`.

## 10. Docker Deployment

Create the environment file:

```bash
cp .env.example .env
```

Build and start:

```bash
docker compose up --build
```

Open:

`http://localhost:8000/docs`

Stop:

```bash
docker compose down
```

The provided Docker configuration runs inference on CPU by default.

## 11. Test Data CSV

The test CSV uses two columns:

- `image`: filename under `data/images/`
- `animals`: one or more supported animal classes separated by `;`

Example:

```csv
image,animals
000000402473.jpg,cat
000000402473.jpg,"cat;dog"
```

The pipeline processes every row in the CSV and writes the combined measurements to `outputs/results.csv`.

## 12. COCO Sample Selection

The sample downloader searches COCO validation annotations for images containing at least two selected animal instances:

```bash
python scripts/download_coco_samples.py --count 5
```

Candidate images can be manually reviewed before use as metrology test cases. The demo focuses on showing the complete segmentation-to-measurement workflow rather than covering every possible animal pose or image quality condition.

## 13. System Architecture

See `docs/architecture.svg`.

## 14. Notes

Model weights are downloaded from Hugging Face on the first inference run and are not included in this repository.

The project reports image-space measurements in pixels. A calibrated camera or reference object is required if physical measurements are required.
