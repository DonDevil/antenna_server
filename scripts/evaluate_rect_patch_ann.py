from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ann.rect_patch_evaluator import evaluate_rect_patch_inverse_ann


def main() -> None:
    summary = evaluate_rect_patch_inverse_ann()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()