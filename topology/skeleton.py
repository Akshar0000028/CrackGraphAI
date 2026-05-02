from __future__ import annotations

import numpy as np
from skimage.morphology import remove_small_objects, skeletonize


def mask_to_skeleton(mask: np.ndarray, min_size: int = 16) -> np.ndarray:
    binary = mask.astype(bool)
    if min_size > 0:
        # Use max_size parameter (newer scikit-image API)
        # To remove objects smaller than min_size, we remove objects of size <= (min_size - 1)
        cleaned = remove_small_objects(binary, min_size=min_size)
    else:
        cleaned = binary
    skel = skeletonize(cleaned)
    return skel.astype(np.uint8)


def connectivity_score(skeleton: np.ndarray) -> float:
    # Ratio of non-isolated skeleton pixels to all skeleton pixels.
    yx = np.argwhere(skeleton > 0)
    if len(yx) == 0:
        return 0.0
    connected = 0
    h, w = skeleton.shape
    for y, x in yx:
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        nb = np.sum(skeleton[y0:y1, x0:x1]) - skeleton[y, x]
        if nb > 0:
            connected += 1
    return connected / len(yx)
