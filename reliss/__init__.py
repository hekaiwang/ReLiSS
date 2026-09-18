"""ReLiSS: missing-modality MRI segmentation."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("RELISS_DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
