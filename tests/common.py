import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "evaluation"))

#: slow tests load the paper checkpoints / models from Hugging Face
SLOW = os.environ.get("MUS3D_SLOW_TESTS") == "1"
#: the released `checkpoints/` directory
CKPTS = Path(os.environ.get("MUS3D_CHECKPOINTS", REPO / "checkpoints"))
CKPT = {
    "seg": CKPTS / "SEG_mixemb",
    "seg_resemb": CKPTS / "SEG_resemb",
    "dit": CKPTS / "DiT",
    "loco": CKPTS / "BS-Locoformer_query",
    "loco_static": CKPTS / "BS-Locoformer_static",
}
