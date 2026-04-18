# Dental ML Project Source Package

This package contains helper modules for the dental pathology detection project.

Suggested notebook workflow:
1. Clone or upload this package.
2. Add `src/` to `sys.path`.
3. Import the helper functions you need.
4. Keep the notebook as the main report and experiment runner.

Example:

```python
import sys
sys.path.append("/kaggle/working/dental_project_src/src")

from data_cleaning import build_clean_dataset
from coco_conversion import convert_polygon_yolo_dataset_to_coco
from detr_utils import prepare_detr_dataset_structure
```
