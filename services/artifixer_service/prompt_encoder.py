"""Prompt / caption embedding helpers.

MVP writes a minimal caption HDF5 when h5py is available; otherwise a JSON sidecar.
Official ArtiFixer embedding is plugged in when ARTIFIXER_ROOT is configured.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_caption_embedding(prompt: str, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import h5py
        import numpy as np

        # Placeholder fixed-size embedding; real encoder replaces this vector.
        vec = np.zeros((1, 4096), dtype=np.float32)
        for i, ch in enumerate(prompt.encode("utf-8")[:64]):
            vec[0, i] = float(ch) / 255.0
        with h5py.File(output_path, "w") as f:
            f.create_dataset("embedding", data=vec)
            f.attrs["prompt"] = prompt
        return output_path
    except Exception:
        sidecar = output_path.with_suffix(".json")
        sidecar.write_text(json.dumps({"prompt": prompt, "embedding": "unavailable"}, indent=2), encoding="utf-8")
        return sidecar
