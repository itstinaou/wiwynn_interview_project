# Implementation Flow

1. Read COCO instances.
2. Keep images containing at least two target animal instances.
3. Detect animal boxes with Grounding DINO.
4. Use each animal box as a SAM 2 prompt and save the animal mask.
5. Crop each animal.
6. Detect `eye` candidates in the animal crop.
7. Use each eye box as a SAM 2 prompt and save the eye mask.
8. Calculate each eye mask centroid.
9. Sort the two eye centers by x coordinate to define image-left/image-right.
10. Calculate per-animal eye distance.
11. Calculate the right-eye distance between any selected pair of animals.
12. Save masks, visualization and CSV.
