from fracture_matching import from_existing_pipeline
from TraditionalSegmentation import get_all_artifact_polygons

img1 = '/home/saybrone/2026-02-25-224552_hyprshot.png'
img2 = '/home/saybrone/2026-02-25-224515_hyprshot.png'

r1 = get_all_artifact_polygons(img1)
r2 = get_all_artifact_polygons(img2)

result = from_existing_pipeline(img1, img2, output_prefix='find_001')
print(result)