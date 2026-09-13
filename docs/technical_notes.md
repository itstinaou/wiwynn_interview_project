# Technical Notes

## Why two model types?

COCO provides animal instance segmentation annotations but not eye annotations. The project therefore separates localization from segmentation:

- Grounding DINO proposes boxes from text prompts.
- SAM 2 turns each box into an individual segmentation mask.

Both the animal and eye final outputs are segmentation masks.

## Eye-to-animal association

Eyes are searched inside each animal crop. The detected eye center is mapped back to the original image and retained for that animal.

## Measurement

The center of each eye mask is used as the measurement point. Euclidean distance is calculated in image coordinates.

If physical measurement is required, the image must be calibrated. A simple planar case can use a known reference scale or homography; a general 3D case requires camera/depth information.

## Verification

Use COCO masks for animal segmentation metrics. Create a small manually annotated eye subset for eye-mask and measurement validation.
